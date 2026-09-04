"""GAT-based predictor for one sampled PPI subgraph."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class LinkAttentionReadout(nn.Module):
    """Target-anchored gated attention over subgraph nodes.

    Faithful mechanism port of RISE-DDI's LinkAttention /
    LinkTransformerLayer (AAAI'25): the two target embeddings act as
    independent queries; every node attends each query with GATv2-style
    gating, with the PPR positional encoding concatenated to the node key:

    .. code-block:: text

        k_p(n)    = softmax_n( a · leaky_relu(W_q(e_p) ⊙ W_r[h_n ; PE(n)]) )
        z_p       = Σ_n k_p(n) · W_r[h_n ; PE(n)]
        z         = dropout(LayerNorm(W_o(e1 + e2 + z_1 + z_2)))

    """

    def __init__(self, dim, pe_dim, heads=1, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.lin_r = nn.Linear(dim + pe_dim, heads * dim)
        self.lin_l = nn.Linear(dim, heads * dim)
        self.att = nn.Parameter(torch.empty(heads, dim))
        nn.init.xavier_uniform_(self.att)
        self.bias = nn.Parameter(torch.zeros(heads * dim))
        self.lin = nn.Linear(heads * dim, dim)
        self.post_att_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e1, e2, h, pe):
        keys = self.lin_r(torch.cat([h, pe], dim=-1))
        keys = keys.view(-1, self.heads, self.dim)
        summaries = []
        for query in (e1, e2):
            gate = self.lin_l(query).view(self.heads, self.dim)
            x = F.leaky_relu(keys * gate.unsqueeze(0), self.negative_slope)
            alpha = (x * self.att).sum(-1)
            alpha = torch.softmax(alpha, dim=0)
            summaries.append(
                (alpha.unsqueeze(-1) * keys).sum(0).reshape(-1) + self.bias
            )
        z = self.lin(e1 + e2 + summaries[0] + summaries[1])
        z = self.post_att_norm(z)
        return self.dropout(z)


class PPIPredictor(nn.Module):
    """Predict the seven PPI relation labels for a sampled subgraph.

    The forward method returns logits; apply ``torch.sigmoid`` for
    probabilities during inference.

    With ``readout="attention"`` the mean-pool readout is kept and augmented
    with a target-anchored attention summary ``z`` (LinkAttention over the
    subgraph nodes with PPR positional encodings, see
    :class:`LinkAttentionReadout`); ``ppr`` must be a :class:`~src.ppr.PPRLookup`
    and ``forward`` then requires ``node_ids`` (the global protein id of every
    feature row) to gather the encodings.
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, num_layers=3, heads=4,
                 dropout=0.1, edge_dim=None, readout="mean", ppr=None):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if edge_dim is not None and edge_dim < 1:
            raise ValueError("edge_dim must be positive when provided")
        if readout not in ("mean", "attention"):
            raise ValueError("readout must be 'mean' or 'attention'")
        if readout == "attention" and ppr is None:
            raise ValueError("readout='attention' requires a PPRLookup")

        self.esm_dim = esm_dim
        self.edge_dim = edge_dim
        self.readout = readout
        self.ppr = ppr
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        conv_kwargs = (
            {"edge_dim": edge_dim, "fill_value": 0.0}
            if edge_dim is not None else {}
        )
        self.convs = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                **conv_kwargs,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        # Common-neighbor and one-hop tiers receive separate PPR positional
        # encoders (RISE-DDI style); nodes adjacent to neither target read 0.
        if readout == "attention":
            self.link_attention = LinkAttentionReadout(hidden_dim, hidden_dim)
            self.ppr_encoder_cn = self._ppr_encoder(hidden_dim)
            self.ppr_encoder_onehop = self._ppr_encoder(hidden_dim)
        output_in = hidden_dim * (4 if readout == "attention" else 3)
        self.output = nn.Sequential(
            nn.Linear(output_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )

    @staticmethod
    def _ppr_encoder(hidden_dim):
        return nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, node_features, edge_index, target_nodes, batch=None,
                edge_attr=None, node_ids=None):
        """Return seven relation logits for one graph.

        ``target_nodes`` contains the two local node indices corresponding to
        the PPI being predicted.  The pair representation uses sum and
        absolute difference, so swapping the endpoints does not change it.
        ``node_ids`` (the global protein id of every feature row) is required
        when ``readout="attention"`` so the PPR positional encodings can be
        gathered for the two targets of each graph.
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
        if self.edge_dim is None:
            if edge_attr is not None:
                raise ValueError("edge_attr requires edge_dim to be configured")
        else:
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_dim is configured")
            edge_attr = torch.as_tensor(
                edge_attr, device=node_features.device, dtype=node_features.dtype
            )
            expected_shape = (edge_index.shape[1], self.edge_dim)
            if edge_attr.ndim != 2 or edge_attr.shape != expected_shape:
                raise ValueError(
                    f"edge_attr must have shape [E, {self.edge_dim}]"
                )
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
            updated = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h + self.dropout(F.elu(updated)))

        u, v = h[target_nodes[:, 0]], h[target_nodes[:, 1]]
        graph = h.new_zeros((int(batch.max()) + 1, h.shape[1]))
        graph.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=graph.shape[0]).to(h.dtype)
        graph = graph / counts.clamp_min(1).unsqueeze(1)
        if self.readout == "attention":
            if node_ids is None:
                raise ValueError(
                    "node_ids is required for the attention readout"
                )
            node_ids = node_ids.to(
                device=node_features.device, dtype=torch.long
            )
            if node_ids.numel() != node_features.shape[0]:
                raise ValueError(
                    "node_ids must contain one global id per feature row"
                )
            z = self._attention_summary(
                h, edge_index, target_nodes, batch, node_ids
            )
            pair = torch.cat((u + v, torch.abs(u - v), graph, z), dim=1)
        else:
            pair = torch.cat((u + v, torch.abs(u - v), graph), dim=1)
        logits = self.output(pair)
        return logits[0] if single_graph else logits

    def _attention_summary(self, h, edge_index, target_nodes, batch, node_ids):
        """Target-anchored attention summary ``z`` for every graph in batch."""
        num_graphs = int(batch.max()) + 1
        batch_list = batch.detach().cpu().tolist()
        edge_list = edge_index.detach().cpu().tolist()
        ids = node_ids.detach().cpu().tolist()

        counts = torch.bincount(batch, minlength=num_graphs).tolist()
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        edges_by_graph = [[] for _ in range(num_graphs)]
        for source, target in zip(edge_list[0], edge_list[1]):
            graph = batch_list[source]
            start = offsets[graph]
            edges_by_graph[graph].append((source - start, target - start))

        summaries = []
        for graph in range(num_graphs):
            start, count = offsets[graph], offsets[graph + 1] - offsets[graph]
            if any(b != graph for b in batch_list[start:start + count]):
                raise ValueError(
                    "batch must list the nodes of each graph contiguously"
                )
            u_row = int(target_nodes[graph, 0])
            v_row = int(target_nodes[graph, 1])
            u_local, v_local = u_row - start, v_row - start
            adjacency = {k: set() for k in range(count)}
            for source, target in edges_by_graph[graph]:
                adjacency[source].add(target)
                adjacency[target].add(source)

            ppr_u = self.ppr.get(ids[u_row])
            ppr_v = self.ppr.get(ids[v_row])
            pe = h.new_zeros(count, h.shape[1])
            cn_rows, cn_values = [], []
            onehop_rows, onehop_values = [], []
            for k in range(count):
                node_global = ids[start + k]
                tier = (k in adjacency[u_local]) + (k in adjacency[v_local])
                if tier == 2:
                    cn_rows.append(k)
                    cn_values.append((
                        ppr_u.get(node_global, 0.0),
                        ppr_v.get(node_global, 0.0),
                    ))
                elif tier == 1:
                    onehop_rows.append(k)
                    onehop_values.append((
                        ppr_u.get(node_global, 0.0),
                        ppr_v.get(node_global, 0.0),
                    ))
            if cn_rows:
                values = torch.tensor(
                    cn_values, device=h.device, dtype=h.dtype
                )
                pe[torch.tensor(cn_rows, device=h.device)] = (
                    self.ppr_encoder_cn(values)
                    + self.ppr_encoder_cn(values.flip(-1))
                )
            if onehop_rows:
                values = torch.tensor(
                    onehop_values, device=h.device, dtype=h.dtype
                )
                pe[torch.tensor(onehop_rows, device=h.device)] = (
                    self.ppr_encoder_onehop(values)
                    + self.ppr_encoder_onehop(values.flip(-1))
                )

            summaries.append(self.link_attention(
                h[u_row], h[v_row], h[start:start + count],
                pe,
            ))
        return torch.stack(summaries)
