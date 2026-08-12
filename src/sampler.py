"""Per-target subgraph sampling for batched PPI training.

Each target pair produces an independent variable-size trajectory; the trainer
combines the resulting graphs for batched predictor computation.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SampledGraph:
    """A sampled subgraph in the coordinate system of ``node_features``."""

    node_index: torch.Tensor
    feature_index: torch.Tensor
    edge_index: torch.Tensor
    target_nodes: torch.Tensor


@dataclass
class SamplingStep:
    """One action and the graph produced after taking that action."""

    graph: SampledGraph
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass
class SamplingTrajectory:
    """Result of sampling a target pair."""

    baseline_graph: SampledGraph
    initial_graph: SampledGraph
    steps: list[SamplingStep]

    @property
    def final_graph(self):
        return self.steps[-1].graph if self.steps else self.initial_graph


class SubgraphSampler(nn.Module):
    """Select a bounded neighborhood around one target PPI pair.

    ``node_features`` and ``edge_index`` describe the current data split.  The
    edge connecting the target pair is excluded in both directions while
    sampling: lazily from a shared immutable split-level adjacency when one is
    supplied, or by removing it from ``edge_index`` when building standalone.
    ``node_index`` maps rows in ``node_features`` to global protein indices;
    when omitted, rows are treated as global indices.
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, max_steps=10,
                 k_hops=3):
        super().__init__()
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if k_hops < 0:
            raise ValueError("k_hops must be non-negative")

        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.k_hops = k_hops
        # State and candidate nodes use independent projections before their
        # pairwise action score is computed.  The first MLP layer receives the
        # concatenated pair, so it can mix state and candidate features.
        self.state_proj = nn.Linear(esm_dim, hidden_dim, bias=False)
        self.neighbor_proj = nn.Linear(esm_dim, hidden_dim, bias=False)
        action_hidden_dim = max(1, hidden_dim // 2)
        self.action_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, action_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(action_hidden_dim, 1),
        )
        for layer in self.action_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.value_head = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None):
        """Sample one trajectory around ``target_nodes``.

        Args:
            node_features: ``[N, esm_dim]`` tensor for the current split.
            edge_index: ``[2, E]`` local edges for the current split.
            target_nodes: two global protein indices ``(u, v)``.
            node_index: optional global id for each feature row.
            training: use stochastic actions when true; defaults to module mode.
            adjacency: optional shared split-level adjacency (``tuple`` of
                ``frozenset``, see :meth:`_build_adjacency`).  When given, the
                target edge is excluded lazily per target; when omitted it is
                built from ``edge_index`` with the target edge already removed.

        Returns:
            ``SamplingTrajectory``.  Each step contains the graph *after* its
            action, while ``log_prob`` and ``value`` describe the decision.
        """
        if training is None:
            training = self.training
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        if adjacency is None:
            safe_edges = self._remove_target_edges(edge_index, target_local)
            adjacency = self._build_adjacency(safe_edges, node_features.shape[0])
        else:
            # ``adjacency`` is a shared immutable split-level structure; exclude
            # the target edge lazily instead of deep-copying every neighbor set
            # (O(E) per target).  Only the two target rows change, so patch the
            # shallow list copy in place.
            u, v = target_local.tolist()
            adjacency = list(adjacency)
            adjacency[u] = adjacency[u] - {v}
            adjacency[v] = adjacency[v] - {u}

        u, v = target_local.tolist()
        selected = [u, v]
        graph_edges = set()
        self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        self._add_real_edges(selected, graph_edges, adjacency)
        allowed_nodes = self._k_hop_region(selected, adjacency, self.k_hops)

        baseline_graph = self._make_graph([u, v], set(), target_local, node_index)
        initial_graph = self._make_graph(selected, graph_edges, target_local, node_index)
        steps = []

        # The frontier is maintained incrementally instead of being rebuilt
        # from every selected node each step: seed it with the initially
        # selected nodes' neighbors (including any proxy), then each step only
        # add the newly chosen node's neighbors.  The k-hop region filter is
        # applied at candidate time.
        selected_set = set(selected)
        frontier = set()
        for node in selected:
            frontier.update(adjacency[node])
        frontier.difference_update(selected_set)

        for _ in range(self.max_steps):
            candidates = sorted(frontier & allowed_nodes)
            if not candidates:
                break

            selected_tensor = torch.tensor(selected, device=node_features.device)
            state = node_features[selected_tensor].to(
                dtype=self.state_proj.weight.dtype
            ).mean(dim=0)
            state_repr = self.state_proj(state)
            candidate_tensor = torch.tensor(candidates, device=node_features.device)
            candidate_repr = self.neighbor_proj(
                node_features[candidate_tensor].to(
                    dtype=self.neighbor_proj.weight.dtype
                )
            )
            state_repr = state_repr.expand(candidate_repr.shape[0], -1)
            attention_input = torch.cat((state_repr, candidate_repr), dim=-1)
            scores = self.action_mlp(attention_input).squeeze(-1)
            probs = torch.softmax(scores, dim=0)
            if training:
                choice = torch.distributions.Categorical(probs).sample()
                log_prob = torch.log(probs[choice])
            else:
                choice = probs.argmax()
                log_prob = probs.new_zeros(())

            action = candidate_tensor[choice]
            action_int = int(action)
            value = self.value_head(state).squeeze(-1)
            self._add_action_edges(action_int, selected, graph_edges, adjacency)
            selected.append(action_int)
            selected_set.add(action_int)
            frontier.discard(action_int)
            frontier.update(adjacency[action_int])
            frontier.difference_update(selected_set)
            graph = self._make_graph(selected, graph_edges, target_local, node_index)
            steps.append(SamplingStep(graph, log_prob, value))

        return SamplingTrajectory(baseline_graph, initial_graph, steps)

    def forward(self, node_features, edge_index, target_nodes, node_index=None,
                training=None, adjacency=None):
        """Alias :meth:`sample` so the sampler follows the PyTorch module API."""
        return self.sample(
            node_features, edge_index, target_nodes, node_index, training, adjacency
        )

    def _prepare_inputs(self, node_features, edge_index, target_nodes, node_index):
        if node_features.ndim != 2 or node_features.shape[1] != self.esm_dim:
            raise ValueError(f"node_features must have shape [N, {self.esm_dim}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if len(target_nodes) != 2:
            raise ValueError("target_nodes must contain exactly two nodes")

        if node_index is None:
            node_index = torch.arange(node_features.shape[0], device=node_features.device)
        else:
            node_index = node_index.to(device=node_features.device, dtype=torch.long)
        target_nodes = torch.as_tensor(
            target_nodes, device=node_features.device, dtype=torch.long
        )
        # ``node_index`` must be strictly increasing: ``PPIGraph.build_graph``
        # emits ``torch.unique`` ids, and the default is ``arange``.  Enforce
        # this contract before using binary search so an unsorted public input
        # cannot be silently mapped to the wrong local nodes.
        if node_index.numel() > 1 and torch.any(node_index[1:] <= node_index[:-1]):
            raise ValueError("node_index must be strictly increasing")
        target_local = torch.searchsorted(node_index, target_nodes)
        if target_local.max() >= node_index.numel():
            raise ValueError("target_nodes must be present in node_index")
        if not torch.equal(node_index[target_local], target_nodes):
            raise ValueError("target_nodes must be present in node_index")
        return node_features, edge_index.to(device=node_features.device), node_index, target_local

    @staticmethod
    def _remove_target_edges(edge_index, target_nodes):
        u, v = target_nodes.tolist()
        keep = ~(
            ((edge_index[0] == u) & (edge_index[1] == v))
            | ((edge_index[0] == v) & (edge_index[1] == u))
        )
        return edge_index[:, keep]

    @staticmethod
    def _build_adjacency(edge_index, num_nodes):
        """Return the undirected adjacency of ``edge_index`` as an immutable
        ``tuple`` of ``frozenset``.

        Immutability lets one structure be built once per split and shared by
        every target; each target excludes its own edge lazily in ``sample``.
        """
        adjacency = [set() for _ in range(num_nodes)]
        for source, target in edge_index.t().tolist():
            if source != target:
                adjacency[source].add(target)
                adjacency[target].add(source)
        return tuple(frozenset(neighbors) for neighbors in adjacency)

    @staticmethod
    def _frontier(selected, adjacency, allowed_nodes=None):
        selected_set = set(selected)
        candidates = {neighbor for node in selected for neighbor in adjacency[node]
                      if neighbor not in selected_set}
        if allowed_nodes is not None:
            candidates.intersection_update(allowed_nodes)
        return sorted(candidates)

    @staticmethod
    def _k_hop_region(seeds, adjacency, k_hops):
        """Return nodes within ``k_hops`` safe-graph hops of ``seeds``."""
        region = set(seeds)
        frontier = set(seeds)
        for _ in range(k_hops):
            next_frontier = {
                neighbor
                for node in frontier
                for neighbor in adjacency[node]
                if neighbor not in region
            }
            region.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return region

    @staticmethod
    def _add_action_edges(action, selected, graph_edges, adjacency):
        for node in selected:
            if action in adjacency[node]:
                graph_edges.add(tuple(sorted((node, action))))

    @staticmethod
    def _add_real_edges(selected, graph_edges, adjacency):
        """Add all safe real edges induced by the initial selected nodes."""
        for index, source in enumerate(selected):
            for target in selected[index + 1:]:
                if target in adjacency[source]:
                    graph_edges.add(tuple(sorted((source, target))))

    def _add_virtual_proxies(self, selected, graph_edges, adjacency,
                             node_features, target_local):
        available = torch.ones(node_features.shape[0], dtype=torch.bool,
                               device=node_features.device)
        available[target_local] = False
        for node in target_local.tolist():
            if adjacency[node] or not available.any():
                continue
            proxy = self._nearest_proxy(node_features[node], node_features, available)
            proxy_int = int(proxy)
            if proxy_int not in selected:
                selected.append(proxy_int)
            graph_edges.add(tuple(sorted((node, proxy_int))))

    @staticmethod
    def _nearest_proxy(node, node_features, available):
        node = F.normalize(node.float(), dim=0)
        candidates = F.normalize(node_features.float(), dim=1)
        scores = candidates @ node
        scores = scores.masked_fill(~available, -torch.inf)
        return scores.argmax()

    @staticmethod
    def _make_graph(selected, graph_edges, target_local, node_index):
        device = node_index.device
        selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
        local = {global_node: i for i, global_node in enumerate(selected)}
        directed_edges = []
        for source, target in sorted(graph_edges):
            directed_edges.extend(((local[source], local[target]),
                                   (local[target], local[source])))
        if directed_edges:
            edge_index = torch.tensor(directed_edges, device=device, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        target_nodes = torch.tensor(
            [local[int(target_local[0])], local[int(target_local[1])]],
            device=device,
            dtype=torch.long,
        )
        return SampledGraph(
            node_index[selected_tensor], selected_tensor, edge_index, target_nodes
        )
