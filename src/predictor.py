"""GAT-based predictor for one sampled PPI subgraph."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class PPIPredictor(nn.Module):
    """Predict the seven PPI relation labels for a sampled subgraph.

    The forward method returns logits.  Call :meth:`predict_proba` for sigmoid
    probabilities used during inference.
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, num_layers=3, heads=4,
                 dropout=0.1):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.esm_dim = esm_dim
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, node_features, edge_index, target_nodes, batch=None):
        """Return seven relation logits for one graph.

        ``target_nodes`` contains the two local node indices corresponding to
        the PPI being predicted.  The pair representation uses sum and
        absolute difference, so swapping the endpoints does not change it.
        """
        if node_features.ndim != 2 or node_features.shape[1] != self.esm_dim:
            raise ValueError(f"node_features must have shape [N, {self.esm_dim}]")
        target_nodes = torch.as_tensor(
            target_nodes, device=node_features.device, dtype=torch.long
        )
        single_graph = target_nodes.ndim == 1
        if target_nodes.ndim == 1:
            if target_nodes.numel() != 2:
                raise ValueError("target_nodes must contain exactly two nodes")
            target_nodes = target_nodes.unsqueeze(0)
        elif target_nodes.ndim != 2 or target_nodes.shape[1] != 2:
            raise ValueError("target_nodes must have shape [2] or [B, 2]")

        node_features = node_features.to(dtype=self.input_proj[0].weight.dtype)
        edge_index = edge_index.to(device=node_features.device, dtype=torch.long)
        if batch is None:
            batch = torch.zeros(
                node_features.shape[0], device=node_features.device, dtype=torch.long
            )
        else:
            batch = batch.to(device=node_features.device, dtype=torch.long)
        if batch.numel() != node_features.shape[0]:
            raise ValueError("batch must contain one graph id per node")

        h = self.input_proj(node_features)
        for conv, norm in zip(self.convs, self.norms):
            updated = conv(h, edge_index)
            h = norm(h + self.dropout(F.elu(updated)))

        u, v = h[target_nodes[:, 0]], h[target_nodes[:, 1]]
        graph = h.new_zeros((int(batch.max()) + 1, h.shape[1]))
        graph.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=graph.shape[0]).to(h.dtype)
        graph = graph / counts.clamp_min(1).unsqueeze(1)
        pair = torch.cat((u + v, torch.abs(u - v), graph), dim=1)
        logits = self.output(pair)
        return logits[0] if single_graph else logits

    def predict_proba(self, node_features, edge_index, target_nodes):
        """Return sigmoid probabilities for inference.

        This compatibility wrapper intentionally mirrors the historical public
        API.  Batched calls should use :meth:`forward` followed by sigmoid when
        a ``batch`` vector is required.
        """
        return torch.sigmoid(self(node_features, edge_index, target_nodes))
