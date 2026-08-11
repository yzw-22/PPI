"""PPIGraph: load a PPI dataset and build per-split graphs + DataLoaders.

Data layout (see dataset/CLAUDE.md):

    PPI Index (0..N-1)
        -> {name}_ppi_list.json           : [[protein_a, protein_b], ...]
    Protein Index (0..M-1)
        -> protein.{name}.sequences.dictionary.tsv  : row i = Protein Index i (id, seq)
        -> {name}_tensor.pt               : tensor[i] = ESM-2 embedding [M, 2560] bfloat16
    Labels (7-dim multi-hot)
        -> protein.actions.{name}.txt     : (item_id_a, item_id_b, mode, ...), OR-aggregated
    Split
        -> {name}_{split}.json            : {"train_index": [...], "val_index": [...], "test_index": [...]}
                                            values are PPI Indices

The 7 label classes are indexed by LABELS order:
    0 reaction, 1 binding, 2 ptmod, 3 activation, 4 inhibition, 5 catalysis, 6 expression
"""

import collections
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class PPIGraph:
    """Container for one PPI dataset (one `name` + one `split`).

    Args:
        name: dataset name, one of ``{'SHS27k', 'SHS148k', 'STRING'}``.
        split: split method. Available splits depend on the dataset
            (``{'SHS27k': ['bfs', 'dfs', 'random'], 'SHS148k': ['dfs', 'random'], 'STRING': ['dfs']}``).
        root: directory holding the dataset files.
        device: where to keep the feature tensor / labels (default ``'cpu'``;
            pass ``'cuda'`` when training on GPU).
        cache_dir: optional directory to cache the built label tensor as
            ``{name}_ppi_labels.pt``, avoiding a re-parse of the large
            ``actions.txt`` on later runs. ``None`` disables caching.
    """

    #: 7 PPI relation classes (index = multi-hot position).
    LABELS = [
        "reaction",
        "binding",
        "ptmod",
        "activation",
        "inhibition",
        "catalysis",
        "expression",
    ]
    LABEL2IDX = {lab: i for i, lab in enumerate(LABELS)}

    #: available split methods per dataset.
    AVAILABLE_SPLITS = {
        "SHS27k": ["bfs", "dfs", "random"],
        "SHS148k": ["dfs", "random"],
        "STRING": ["dfs"],
    }
    SPLIT_NAMES = ["train", "val", "test"]

    def __init__(self, name="SHS27k", split="bfs", root="dataset",
                 device="cpu", cache_dir=None):
        if name not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"Unknown dataset {name!r}; expected one of {sorted(self.AVAILABLE_SPLITS)}"
            )
        if split not in self.AVAILABLE_SPLITS[name]:
            raise ValueError(
                f"Split {split!r} not available for {name}; "
                f"expected one of {self.AVAILABLE_SPLITS[name]}"
            )
        self.name = name
        self.split = split
        self.root = Path(root)
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self):
        """Load every file needed by the dataset and precompute labels."""
        self.tensor = torch.load(
            self.root / f"{self.name}_tensor.pt", weights_only=True
        ).to(self.device)
        self.ppi_list = json.loads(
            (self.root / f"{self.name}_ppi_list.json").read_text()
        )  # list of [protein_a, protein_b]

        with open(self.root / f"protein.{self.name}.sequences.dictionary.tsv") as f:
            self.proteins = list(csv.DictReader(f, delimiter="\t"))
        if len(self.proteins) != self.tensor.shape[0]:
            raise RuntimeError(
                f"{self.name}: protein dict ({len(self.proteins)}) rows != "
                f"tensor rows ({self.tensor.shape[0]})"
            )

        self.split_index = json.loads(
            (self.root / f"{self.name}_{self.split}.json").read_text()
        )

        self.ppi_labels = self._load_labels()
        self.ppi = torch.tensor(self.ppi_list, dtype=torch.long, device=self.device)
        # ppi[a, 0] / ppi[a, 1] = protein indices of PPI index a

    def _load_labels(self):
        """Return a ``[n_ppi, 7]`` float32 multi-hot label tensor.

        Labels come from ``actions.txt``: OR-aggregate ``mode`` over all rows of
        an (undirected) protein pair. Cache the result in ``cache_dir`` when set.
        """
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{self.name}_ppi_labels.pt"
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True).to(self.device)

        pair_modes = collections.defaultdict(set)
        with open(self.root / f"protein.actions.{self.name}.txt") as f:
            header = f.readline().rstrip("\n").split("\t")
            col = {c: header.index(c) for c in ("item_id_a", "item_id_b", "mode")}
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:  # skip blank / malformed rows
                    continue
                a, b, mode = parts[col["item_id_a"]], parts[col["item_id_b"]], parts[col["mode"]]
                if mode not in self.LABEL2IDX:
                    continue
                key = (a, b) if a <= b else (b, a)  # undirected pair key
                pair_modes[key].add(self.LABEL2IDX[mode])

        labels = torch.zeros(len(self.ppi_list), len(self.LABELS), dtype=torch.float32)
        for ppi_idx, (ia, ib) in enumerate(self.ppi_list):
            key = (self.proteins[ia]["id"], self.proteins[ib]["id"])
            if key[0] > key[1]:
                key = (key[1], key[0])
            for cls_idx in pair_modes.get(key, ()):
                labels[ppi_idx, cls_idx] = 1.0

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(labels.cpu(), cache_path)

        return labels.to(self.device)

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def get_ppi_indices(self, split_name="train"):
        """Return the PPI indices belonging to ``split_name`` as a LongTensor."""
        if split_name not in self.SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {self.SPLIT_NAMES}")
        return torch.tensor(
            self.split_index[f"{split_name}_index"], dtype=torch.long, device=self.device
        )

    def build_graph(self, split_name="train", undirected=True):
        """Build the PPI graph induced by one split.

        ``ppi_list`` lists every undirected PPI pair exactly once, in an
        arbitrary direction; with ``undirected=True`` (default) the reverse
        edges are added so message passing sees a true undirected graph.

        Returns a dict with:
            edge_index  : LongTensor [2, E]   local node indices of each edge (u -> v)
            node_index  : LongTensor [N]      global protein indices of the graph's nodes
            node_feat   : Tensor   [N, 2560]  ESM-2 embeddings of the nodes

        Per-ppi multi-hot labels are not included here; the trainer reads them
        directly from ``ppi_labels`` (see ``run`` in ``train_shs27k``).

        ``node_index`` is sorted and ``edge_index`` always refers to *local*
        node ids (0..N-1).  The graph and its feature/proxy candidate pool
        therefore contain only nodes present in the requested split.
        """
        ppi_idx = self.get_ppi_indices(split_name)
        u = self.ppi[ppi_idx, 0]
        v = self.ppi[ppi_idx, 1]

        # Each undirected PPI pair appears once in ppi_list; add the reverse
        # edge (v, u) so the graph is undirected for message passing.
        if undirected:
            u_orig, v_orig = u, v
            u = torch.cat([u_orig, v_orig])
            v = torch.cat([v_orig, u_orig])

        nodes = torch.unique(torch.cat([u, v]))
        node_index = nodes
        # global protein index -> local node id
        local = torch.full(
            (self.tensor.shape[0],), -1, dtype=torch.long, device=self.device
        )
        local[nodes] = torch.arange(nodes.numel(), device=self.device)
        edge_index = torch.stack([local[u], local[v]], dim=0)  # [2, E]
        node_feat = self.tensor[nodes]  # [N, 2560]

        return {
            "edge_index": edge_index,
            "node_index": node_index,
            "node_feat": node_feat,
        }

    # ------------------------------------------------------------------ #
    # DataLoader
    # ------------------------------------------------------------------ #
    def get_dataloader(self, split_name="train", batch_size=32, shuffle=None,
                       num_workers=0, drop_last=False, pin_memory=False,
                       collate_fn=None, **kwargs):
        """Build a DataLoader over the PPI pairs of ``split_name``.

        Each sample is a ``(u, v, label)`` tuple:
            u, v   : LongTensor [B]  protein indices of the interacting pair
            label  : FloatTensor [B, 7]  multi-hot label

        ``shuffle`` defaults to ``True`` for train, ``False`` otherwise.
        """
        if self.device.type == "cuda" and num_workers:
            raise ValueError("num_workers must be 0 when PPIGraph uses a CUDA device")
        if self.device.type == "cuda" and pin_memory:
            raise ValueError("pin_memory requires CPU tensors; use device='cpu'")
        if shuffle is None:
            shuffle = split_name == "train"
        dataset = _PPIDataset(self, self.get_ppi_indices(split_name))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def __repr__(self):
        sizes = {k: len(self.split_index[f"{k}_index"])
                 for k in self.SPLIT_NAMES}
        return (f"PPIGraph(name={self.name!r}, split={self.split!r}, "
                f"proteins={self.tensor.shape[0]}, dim={self.tensor.shape[1]}, "
                f"n_ppi={len(self.ppi_list)}, splits={sizes})")


class _PPIDataset(Dataset):
    """Dataset over PPI indices; each item yields ``(u, v, label)``."""

    def __init__(self, graph, ppi_indices):
        self.graph = graph
        self.ppi_indices = ppi_indices

    def __len__(self):
        return self.ppi_indices.numel()

    def __getitem__(self, i):
        ppi_idx = int(self.ppi_indices[i])
        u, v = self.graph.ppi[ppi_idx, 0], self.graph.ppi[ppi_idx, 1]
        label = self.graph.ppi_labels[ppi_idx]
        return u, v, label
