"""Per-target subgraph sampling for batched PPI training.

Each target pair produces an independent variable-size trajectory; the trainer
combines the resulting graphs for batched predictor computation.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _TargetSafeAdjacency:
    """Lazy view of split adjacency with one target edge overlaid."""

    def __init__(self, base, source, target):
        self.base = base
        self.source = int(source)
        self.target = int(target)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        neighbors = self.base[index]
        if index == self.source:
            return neighbors - {self.target}
        if index == self.target:
            return neighbors - {self.source}
        return neighbors


@dataclass
class SampledGraph:
    """A sampled subgraph in the coordinate system of ``node_features``."""

    node_index: torch.Tensor
    feature_index: torch.Tensor
    edge_index: torch.Tensor
    target_nodes: torch.Tensor
    # Diagnostics are optional so existing callers can still construct graphs
    # positionally.  ``proxy_nodes`` are split-local ids.
    proxy_nodes: tuple = ()
    real_edge_count: int = 0


@dataclass
class SamplingStep:
    """One action and the graph produced after taking that action."""

    graph: SampledGraph
    log_prob: torch.Tensor
    value: torch.Tensor
    is_stop: bool = False


@dataclass
class SamplingTrajectory:
    """Result of sampling a target pair."""

    baseline_graph: SampledGraph
    steps: list[SamplingStep]
    proxy_nodes: tuple = ()
    stopped: bool = False

    @property
    def final_graph(self):
        return self.steps[-1].graph if self.steps else self.baseline_graph

    @property
    def action_count(self):
        return len(self.steps)

    @property
    def context_node_count(self):
        # Baseline target_nodes are local graph positions; use graph size and
        # the known proxy set for a stable, model-independent diagnostic.
        return max(0, int(self.final_graph.node_index.numel()) - 2 - len(self.proxy_nodes))


class SubgraphSampler(nn.Module):
    """Select a bounded subgraph around one target PPI pair.

    ``node_features`` and ``edge_index`` describe the current data split.  The
    edge connecting the target pair is excluded in both directions while
    sampling: lazily from a shared immutable split-level adjacency when one is
    supplied, or by removing it from ``edge_index`` when building standalone.
    ``node_index`` maps rows in ``node_features`` to global protein indices;
    when omitted, rows are treated as global indices.

    ``fixed_num`` limits the number of deterministic, seed-42 sampled
    one-hop context nodes selected independently for each initial seed in
    ``{u, v, proxy}``.  ``max_steps`` limits the number of policy actions.
    The resulting graph size depends on safe adjacency, proxy selection,
    duplicate removal, and frontier availability; there is no aggregate
    context-node budget derived from these parameters.
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, max_steps=10,
                 fixed_num=1, complexity_penalty=0.0):
        super().__init__()
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if fixed_num < 0:
            raise ValueError("fixed_num must be non-negative")

        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.fixed_num = fixed_num
        self.complexity_penalty = float(complexity_penalty)
        self._normalized_feature_cache = {}
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
        # A separate STOP logit makes termination an explicit policy action.
        self.stop_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.stop_head.bias, -2.0)

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
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )

        u, v = target_local.tolist()
        selected = [u, v]
        graph_edges = set()
        proxy_nodes = self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        selected.extend(self._sample_initial_neighbors(selected, adjacency))
        # G0 is the sole baseline: include every safe split-local edge induced
        # by its nodes, in addition to any virtual proxy edges.
        self._add_real_edges(selected, graph_edges, adjacency)
        baseline_graph = self._make_graph(
            selected, graph_edges, target_local, node_index, proxy_nodes
        )
        steps = []

        # The frontier is maintained incrementally instead of being rebuilt
        # from every selected node each step: seed it with the initially
        # selected nodes' neighbors (including any proxy), then each step only
        # add the newly chosen node's neighbors.
        selected_set = set(selected)
        frontier = set()
        for node in selected:
            frontier.update(adjacency[node])
        frontier.difference_update(selected_set)

        for _ in range(self.max_steps):
            candidates = sorted(frontier)
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
            stop_logit = self.stop_head(state_repr[:1]).squeeze()
            logits = torch.cat((scores, stop_logit.reshape(1)))
            probs = torch.softmax(logits, dim=0)
            if training:
                choice = torch.distributions.Categorical(probs).sample()
                log_prob = torch.log(probs[choice])
            else:
                choice = probs.argmax()
                log_prob = probs.new_zeros(())

            value = self.value_head(state).squeeze(-1)
            if int(choice) == len(candidates):
                graph = self._make_graph(
                    selected, graph_edges, target_local, node_index, proxy_nodes
                )
                steps.append(SamplingStep(graph, log_prob, value, is_stop=True))
                return SamplingTrajectory(
                    baseline_graph, steps, tuple(sorted(proxy_nodes)), stopped=True
                )
            action = candidate_tensor[choice]
            action_int = int(action)
            self._add_action_edges(action_int, selected, graph_edges, adjacency)
            selected.append(action_int)
            selected_set.add(action_int)
            frontier.discard(action_int)
            frontier.update(adjacency[action_int])
            frontier.difference_update(selected_set)
            graph = self._make_graph(
                selected, graph_edges, target_local, node_index, proxy_nodes
            )
            steps.append(SamplingStep(graph, log_prob, value))

        return SamplingTrajectory(
            baseline_graph, steps, tuple(sorted(proxy_nodes)), stopped=False
        )

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

    def _prepare_adjacency(self, edge_index, target_local, num_nodes, adjacency):
        if adjacency is None:
            safe_edges = self._remove_target_edges(edge_index, target_local)
            return self._build_adjacency(safe_edges, num_nodes)

        # ``adjacency`` is a shared immutable split-level structure; exclude
        # the target edge lazily instead of deep-copying every neighbor set.
        u, v = target_local.tolist()
        return _TargetSafeAdjacency(adjacency, u, v)

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

    def _sample_initial_neighbors(self, seed_nodes, adjacency):
        """Sample at most ``fixed_num`` neighbors independently per seed.

        A private generator seeded with 42 makes G0 reproducible without
        resetting the global RNG used for RL action sampling.
        """
        generator = torch.Generator(device="cpu").manual_seed(42)
        initial_set = set(seed_nodes)
        sampled = set()
        for seed in seed_nodes:
            candidates = sorted(
                neighbor for neighbor in adjacency[seed]
                if neighbor not in initial_set
            )
            if len(candidates) > self.fixed_num:
                positions = torch.randperm(
                    len(candidates), generator=generator
                )[:self.fixed_num].tolist()
                candidates = [candidates[position] for position in positions]
            sampled.update(candidates)
        return sorted(sampled)

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
        proxy_nodes = set()
        cache_key = (node_features.data_ptr(), tuple(node_features.shape),
                     str(node_features.device), getattr(node_features, "_version", 0))
        normalized = self._normalized_feature_cache.get(cache_key)
        if normalized is None or normalized.device != node_features.device:
            normalized = F.normalize(node_features.float(), dim=1)
            self._normalized_feature_cache[cache_key] = normalized
        available = torch.ones(node_features.shape[0], dtype=torch.bool,
                               device=node_features.device)
        available[target_local] = False
        for node in target_local.tolist():
            if adjacency[node] or not available.any():
                continue
            proxy = self._nearest_proxy_normalized(normalized[node], normalized, available)
            proxy_int = int(proxy)
            if proxy_int not in selected:
                selected.append(proxy_int)
            proxy_nodes.add(proxy_int)
            graph_edges.add(tuple(sorted((node, proxy_int))))
        return proxy_nodes

    @staticmethod
    def _nearest_proxy(node, node_features, available):
        node = F.normalize(node.float(), dim=0)
        candidates = F.normalize(node_features.float(), dim=1)
        return SubgraphSampler._nearest_proxy_normalized(node, candidates, available)

    @staticmethod
    def _nearest_proxy_normalized(node, node_features, available):
        scores = node @ node_features.T
        scores = scores.masked_fill(~available, -torch.inf)
        return scores.argmax()

    @staticmethod
    def _make_graph(selected, graph_edges, target_local, node_index, proxy_nodes=()):
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
        target_set = {int(target_local[0]), int(target_local[1])}
        real_edges = sum(
            not ((source in proxy_nodes and target in target_set)
                 or (target in proxy_nodes and source in target_set))
            for source, target in graph_edges
        )
        return SampledGraph(
            node_index[selected_tensor], selected_tensor, edge_index, target_nodes,
            tuple(sorted(proxy_nodes)), int(real_edges)
        )


class RandomSubgraphSampler(SubgraphSampler):
    """Build a fixed random one-hop subgraph for sampler ablations.

    The target edge is removed exactly as in :class:`SubgraphSampler`.  A
    deterministic embedding-nearest proxy is added when a target has no safe
    neighbors, then at most ``max_context_nodes`` nodes are sampled without
    replacement from the union of the safe one-hop neighborhoods of
    ``{u, v, proxy}``.  The private seed is reset for each target so repeated
    calls produce the same graph and do not affect the training RNG.
    """

    def __init__(self, esm_dim=2560, max_context_nodes=10, random_seed=42,
                 use_proxy=True):
        nn.Module.__init__(self)
        if max_context_nodes < 0:
            raise ValueError("max_context_nodes must be non-negative")
        self.esm_dim = esm_dim
        self.max_context_nodes = max_context_nodes
        self.random_seed = random_seed
        self.use_proxy = bool(use_proxy)
        self._normalized_feature_cache = {}

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None):
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )

        selected = target_local.tolist()
        graph_edges = set()
        proxy_nodes = set()
        if self.use_proxy:
            proxy_nodes = self._add_virtual_proxies(
                selected, graph_edges, adjacency, node_features, target_local
            )
        selected_set = set(selected)
        candidates = sorted({
            neighbor
            for node in selected
            for neighbor in adjacency[node]
            if neighbor not in selected_set
        })
        if len(candidates) > self.max_context_nodes:
            generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
            positions = torch.randperm(
                len(candidates), generator=generator
            )[:self.max_context_nodes].tolist()
            candidates = [candidates[position] for position in positions]
        selected.extend(candidates)
        self._add_real_edges(selected, graph_edges, adjacency)
        graph = self._make_graph(
            selected, graph_edges, target_local, node_index, proxy_nodes
        )
        return SamplingTrajectory(graph, [], tuple(sorted(proxy_nodes)), stopped=False)

    def forward(self, node_features, edge_index, target_nodes, node_index=None,
                training=None, adjacency=None):
        return self.sample(
            node_features, edge_index, target_nodes, node_index, training, adjacency
        )


class RandomIterativeSubgraphSampler(SubgraphSampler):
    """Expand the learned sampler's G0 with deterministic random actions.

    The initial graph and frontier are identical to :class:`SubgraphSampler`,
    but each action selects one current frontier node using a private seed-42
    generator.  This module deliberately has no learnable parameters: its
    trajectories are used only for the Predictor update in ablation runs.
    """

    def __init__(self, esm_dim=2560, fixed_num=1, max_steps=10,
                 random_seed=42):
        nn.Module.__init__(self)
        if fixed_num < 0:
            raise ValueError("fixed_num must be non-negative")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        self.esm_dim = esm_dim
        self.fixed_num = fixed_num
        self.max_steps = max_steps
        self.random_seed = random_seed
        self._normalized_feature_cache = {}

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None):
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )

        selected = target_local.tolist()
        graph_edges = set()
        proxy_nodes = self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        selected.extend(self._sample_initial_neighbors(selected, adjacency))
        self._add_real_edges(selected, graph_edges, adjacency)
        baseline_graph = self._make_graph(
            selected, graph_edges, target_local, node_index, proxy_nodes
        )

        selected_set = set(selected)
        frontier = set()
        for node in selected:
            frontier.update(adjacency[node])
        frontier.difference_update(selected_set)

        generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
        zero = node_features.new_zeros(())
        steps = []
        for _ in range(self.max_steps):
            candidates = sorted(frontier)
            if not candidates:
                break
            position = int(torch.randperm(
                len(candidates), generator=generator
            )[0])
            action_int = candidates[position]
            self._add_action_edges(action_int, selected, graph_edges, adjacency)
            selected.append(action_int)
            selected_set.add(action_int)
            frontier.discard(action_int)
            frontier.update(adjacency[action_int])
            frontier.difference_update(selected_set)
            graph = self._make_graph(
                selected, graph_edges, target_local, node_index, proxy_nodes
            )
            steps.append(SamplingStep(graph, zero, zero))

        return SamplingTrajectory(
            baseline_graph, steps, tuple(sorted(proxy_nodes)), stopped=False
        )

    def forward(self, node_features, edge_index, target_nodes, node_index=None,
                training=None, adjacency=None):
        return self.sample(
            node_features, edge_index, target_nodes, node_index, training, adjacency
        )
