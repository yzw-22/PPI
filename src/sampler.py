"""Per-target subgraph sampling for batched PPI training.

Each target pair produces an independent variable-size trajectory; the trainer
combines the resulting graphs for batched predictor computation.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _TargetSafeAdjacency:
    """Read-only adjacency view with one target edge removed.

    The split-level adjacency remains shared. Only the two target rows are
    materialized once, avoiding an O(N) list copy for every trajectory.
    """

    def __init__(self, base, source, target):
        self.base = base
        self.source = int(source)
        self.target = int(target)
        self.source_neighbors = base[self.source] - {self.target}
        self.target_neighbors = base[self.target] - {self.source}

    def __getitem__(self, index):
        if index == self.source:
            return self.source_neighbors
        if index == self.target:
            return self.target_neighbors
        return self.base[index]


@dataclass
class SampledGraph:
    """A sampled subgraph in the coordinate system of ``node_features``."""

    node_index: torch.Tensor
    feature_index: torch.Tensor
    edge_index: torch.Tensor
    target_nodes: torch.Tensor
    edge_attr: torch.Tensor | None = None


@dataclass(frozen=True)
class EdgeRelationLookup:
    """Sparse relation lookup containing visible undirected edges only.

    The training entry inserts only train-split edges. Missing entries therefore
    cover validation/test topology and virtual proxy edges and map to zeros.
    """

    keys: torch.Tensor
    values: torch.Tensor
    num_nodes: int

    @classmethod
    def from_pairs(cls, pairs, values, num_nodes):
        values = torch.as_tensor(values, dtype=torch.float32)
        pairs = torch.as_tensor(pairs, dtype=torch.long, device=values.device)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [E, 2]")
        if values.ndim != 2 or values.shape[0] != pairs.shape[0]:
            raise ValueError("values must have shape [E, relation_dim]")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if pairs.numel() and (pairs.min() < 0 or pairs.max() >= num_nodes):
            raise ValueError("pairs contain an out-of-range node")

        source = torch.minimum(pairs[:, 0], pairs[:, 1])
        target = torch.maximum(pairs[:, 0], pairs[:, 1])
        keys = source * num_nodes + target
        keys, order = torch.sort(keys)
        values = values[order]
        if keys.numel() > 1 and torch.any(keys[1:] == keys[:-1]):
            raise ValueError("pairs must contain unique undirected edges")
        return cls(keys, values, int(num_nodes))

    @property
    def relation_dim(self):
        return self.values.shape[1]

    def lookup(self, pairs):
        """Return relation rows for local-id pairs; unknown edges are zero."""
        pairs = torch.as_tensor(pairs, dtype=torch.long, device=self.keys.device)
        if pairs.numel() == 0:
            return self.values.new_zeros((0, self.relation_dim))
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [E, 2]")
        source = torch.minimum(pairs[:, 0], pairs[:, 1])
        target = torch.maximum(pairs[:, 0], pairs[:, 1])
        query = source * self.num_nodes + target
        result = self.values.new_zeros((pairs.shape[0], self.relation_dim))
        # With no stored keys every query is a miss and stays zero; the empty
        # lookup is never indexed (its keys tensor cannot be subscripted).
        if self.keys.numel():
            positions = torch.searchsorted(self.keys, query)
            matched = positions < self.keys.numel()
            safe_positions = positions.clamp_max(self.keys.numel() - 1)
            matched &= self.keys[safe_positions] == query
            result[matched] = self.values[safe_positions[matched]]
        return result


@dataclass
class SamplingStep:
    """One action and the graph produced after taking that action."""

    graph: SampledGraph
    log_prob: torch.Tensor


@dataclass
class SamplingTrajectory:
    """Result of sampling a target pair."""

    baseline_graph: SampledGraph
    steps: list[SamplingStep]

    @property
    def final_graph(self):
        return self.steps[-1].graph if self.steps else self.baseline_graph


class SubgraphSampler(nn.Module):
    """Select a bounded subgraph around one target PPI pair.

    ``node_features`` and ``edge_index`` describe the shared knowledge graph
    (the full dataset graph in the training entry).  The edge connecting the
    target pair is excluded in both directions while sampling: lazily from a
    shared immutable graph-level adjacency when one is supplied, or by
    removing it from ``edge_index`` when building standalone.
    ``node_index`` maps rows in ``node_features`` to global protein indices;
    when omitted, rows are treated as global indices.

    The baseline graph contains only the target pair and any required virtual
    proxy. Candidate expansions are restricted to their safe ``k_hops``
    neighborhood.

    With ``structural_features=True`` the action score additionally receives
    an 8-D topology block per candidate (common-neighbor, target-touching,
    degree, selected-connectivity, target distances, Adamic-Adar), injected
    through a zero-initialized projection plus a linear skip channel that is
    initialized to the ``HeuristicSampler`` ranking (see
    ``STRUCTURAL_PRIOR``): greedy evaluation starts at the heuristic policy
    and REINFORCE refines both paths from there.
    """

    #: Width of the structural candidate-feature block; the column layout is
    #: documented on :meth:`_topo_features`.
    STRUCTURAL_FEATURES = 8
    #: Heuristic prior for the structural skip channel ``scores += w·φ_topo``.
    #: (8, 4, 1, 0, …) reproduces the ``HeuristicSampler`` ranking under
    #: greedy argmax — common neighbors score 8 + 2·4, single-target neighbors
    #: 4, the rest only their degree column — while softmax sampling keeps an
    #: e^-8 exploration floor.  The vector stays trainable: REINFORCE starts
    #: at the heuristic and refines from there.
    STRUCTURAL_PRIOR = (8.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(self, esm_dim=2560, hidden_dim=512, max_steps=10,
                 k_hops=1, relation_dim=None, structural_features=False):
        super().__init__()
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if k_hops < 0:
            raise ValueError("k_hops must be non-negative")
        if relation_dim is not None and relation_dim <= 0:
            raise ValueError("relation_dim must be positive when provided")

        self.esm_dim = esm_dim
        self.max_steps = max_steps
        self.k_hops = k_hops
        self.relation_dim = relation_dim
        self.structural_features = structural_features
        # Shared-adjacency degree-scale cache for the structural features;
        # populated on first use (see ``_degree_scale``).
        self._degree_cache = None
        # State and candidate nodes use independent projections before their
        # pairwise action score is computed.
        self.state_proj = nn.Linear(esm_dim, hidden_dim, bias=False)
        self.neighbor_proj = nn.Linear(esm_dim, hidden_dim, bias=False)
        # The pair representation is mapped back to hidden_dim and the
        # projected state is added back (residual-style), then scored by a
        # LayerNorm/Tanh head.  The first MLP layer mixes state and candidate
        # features; the residual keeps the state contribution explicit.
        action_hidden_dim = max(1, hidden_dim // 2)
        self.pair_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
            nn.Tanh(),
            nn.Linear(action_hidden_dim, 1),
        )
        for layer in (self.pair_proj, *self.fc):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.relation_proj = None
        if relation_dim is not None:
            # Do not perturb initialization of downstream modules (notably the
            # Predictor) when enabling this optional branch. This keeps seeded
            # relation ablations paired on all pre-existing parameters.
            with torch.random.fork_rng(devices=[]):
                self.relation_proj = nn.Linear(
                    relation_dim, hidden_dim, bias=False
                )
                nn.init.xavier_uniform_(self.relation_proj.weight)
        if structural_features:
            # Zero-initialized projection: the learned structural path starts
            # as a no-op, so the prior-initialized skip below alone defines
            # the initial policy.  The constructor runs under fork_rng and
            # zeros_ consumes no RNG, keeping seeded runs paired with the
            # flag off (same property as the relation branch).
            with torch.random.fork_rng(devices=[]):
                self.struct_proj = nn.Linear(self.STRUCTURAL_FEATURES, hidden_dim)
            nn.init.zeros_(self.struct_proj.weight)
            nn.init.zeros_(self.struct_proj.bias)
            self.struct_prior = nn.Parameter(
                torch.tensor(self.STRUCTURAL_PRIOR)
            )

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None, edge_relations=None):
        """Sample one trajectory around ``target_nodes``.

        Args:
            node_features: ``[N, esm_dim]`` tensor for the shared knowledge graph.
            edge_index: ``[2, E]`` local edges for the shared knowledge graph.
            target_nodes: two global protein indices ``(u, v)``.
            node_index: optional global id for each feature row.
            training: use stochastic actions when true; defaults to module mode.
            adjacency: optional shared graph-level adjacency (``tuple`` of
                ``frozenset``, see :meth:`_build_adjacency`).  When given, the
                target edge is excluded lazily per target; when omitted it is
                built from ``edge_index`` with the target edge already removed.
            edge_relations: optional visible-edge relation lookup. Relations
                absent from the lookup and virtual edges receive all-zero rows.

        Returns:
            ``SamplingTrajectory``.  Each step contains the graph *after* its
            action, while ``log_prob`` describes the decision.
        """
        if training is None:
            training = self.training
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )
        if self.relation_dim is not None:
            if edge_relations is None:
                raise ValueError(
                    "edge_relations are required for a relation-aware sampler"
                )
            if edge_relations.relation_dim != self.relation_dim:
                raise ValueError(
                    f"edge_relations must have dimension {self.relation_dim}"
                )

        u, v = target_local.tolist()
        selected = [u, v]
        graph_edges = set()
        self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        # G0 is the sole baseline: it contains only targets and any virtual
        # proxies. Safe one-hop neighbors are introduced by frontier actions.
        self._add_real_edges(selected, graph_edges, adjacency)
        baseline_graph = self._make_graph(
            selected, graph_edges, target_local, node_index, edge_relations
        )
        allowed_nodes = self._k_hop_region(selected, adjacency, self.k_hops)

        # Structural features: distances to each target are fixed for the
        # whole trajectory (the safe adjacency never changes), so the two
        # bounded BFS runs happen once; only the selected-connectivity column
        # is recomputed per step.
        dist_u = dist_v = None
        log_dmax = None
        if self.structural_features:
            base_adjacency = (
                adjacency.base
                if isinstance(adjacency, _TargetSafeAdjacency)
                else adjacency
            )
            log_dmax = self._degree_scale(base_adjacency)
            distance_cap = 2 * self.k_hops
            dist_u = self._bounded_distances(
                u, allowed_nodes, adjacency, distance_cap
            )
            dist_v = self._bounded_distances(
                v, allowed_nodes, adjacency, distance_cap
            )

        # ``allowed_nodes`` is computed once per trajectory. The frontier is
        # then maintained incrementally without repeating the k-hop traversal.
        selected_set = set(selected)
        frontier = {
            neighbor
            for node in selected
            for neighbor in adjacency[node]
            if neighbor in allowed_nodes
        }
        frontier.difference_update(selected_set)

        # Step graphs are snapshotted incrementally and materialized together
        # after the trajectory: the decision loop stays light and the shared
        # ``local`` id map is extended rather than rebuilt per step.
        step_records = []
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
            if self.relation_proj is not None:
                candidate_relations = self._candidate_relation_features(
                    candidates, selected, adjacency, edge_relations
                ).to(
                    device=candidate_repr.device,
                    dtype=self.relation_proj.weight.dtype,
                )
                candidate_repr = (
                    candidate_repr + self.relation_proj(candidate_relations)
                )
            topo = None
            if self.structural_features:
                topo = self._topo_features(
                    candidates, u, v, selected_set, adjacency,
                    dist_u, dist_v, self.k_hops, log_dmax,
                )
                candidate_repr = candidate_repr + self.struct_proj(
                    topo.to(
                        device=candidate_repr.device,
                        dtype=self.struct_proj.weight.dtype,
                    )
                )
            state_repr = state_repr.expand(candidate_repr.shape[0], -1)
            attention_input = torch.cat((state_repr, candidate_repr), dim=-1)
            # Residual-style pair mixing: map the concatenated pair back to
            # hidden_dim and add the projected state back to every row.
            mapped = self.pair_proj(attention_input)
            h = mapped + state_repr
            scores = self.fc(h).squeeze(-1)
            if self.structural_features:
                # Heuristic-prior skip channel; ``struct_prior`` stays
                # trainable so REINFORCE refines the ranking from the
                # heuristic initialization.
                scores = scores + topo.to(scores.dtype) @ self.struct_prior.to(
                    device=scores.device, dtype=scores.dtype
                )
            probs = torch.softmax(scores, dim=0)
            if training:
                choice = torch.distributions.Categorical(probs).sample()
                log_prob = torch.log(probs[choice])
            else:
                choice = probs.argmax()
                log_prob = probs.new_zeros(())

            action = candidate_tensor[choice]
            action_int = int(action)
            self._add_action_edges(action_int, selected, graph_edges, adjacency)
            selected.append(action_int)
            selected_set.add(action_int)
            # No ``discard`` needed: the difference_update below removes every
            # selected node (including ``action_int``) from the frontier.
            frontier.update(
                neighbor for neighbor in adjacency[action_int]
                if neighbor in allowed_nodes
            )
            frontier.difference_update(selected_set)
            step_records.append((log_prob, selected.copy(), graph_edges.copy()))

        step_graphs = self._make_graphs(
            [(selected, edges) for _, selected, edges in step_records],
            target_local, node_index, edge_relations,
        )
        steps = [
            SamplingStep(graph, log_prob)
            for (log_prob, _, _), graph in zip(step_records, step_graphs)
        ]
        return SamplingTrajectory(baseline_graph, steps)

    def _candidate_relation_features(self, candidates, selected, adjacency,
                                     edge_relations):
        """Aggregate visible candidate-to-state relations with multi-hot OR.

        ``edge_relations`` is validated by :meth:`sample` before this helper
        runs.  Only edges present in the target-safe adjacency are queried, so
        the current target edge can never contribute. Unknown held-out and
        virtual edges are represented by the lookup's all-zero fallback.
        """
        relation_dim = edge_relations.relation_dim
        result = edge_relations.values.new_zeros(
            (len(candidates), relation_dim)
        )
        pairs = []
        owners = []
        for candidate_index, candidate in enumerate(candidates):
            for node in selected:
                if candidate in adjacency[node]:
                    pairs.append((node, candidate))
                    owners.append(candidate_index)
        if not pairs:
            return result

        values = edge_relations.lookup(pairs)
        # A candidate may touch several selected nodes and each PPI can itself
        # be multi-label. Element-wise max is therefore the logical OR over all
        # visible incident relation types.
        for candidate_index in range(len(candidates)):
            positions = [
                index for index, owner in enumerate(owners)
                if owner == candidate_index
            ]
            if positions:
                result[candidate_index] = values[positions].amax(dim=0)
        return result

    def _degree_scale(self, base_adjacency):
        """Return ``log1p(max degree)`` of the shared base adjacency.

        Cached per adjacency object; holding the reference keeps the identity
        (and therefore the cache hit) stable for the lifetime of the graph, so
        the training entry's shared adjacency is scanned exactly once.
        """
        cached = self._degree_cache
        if cached is not None and cached[0] is base_adjacency:
            return cached[1]
        if len(base_adjacency):
            log_dmax = math.log1p(max(len(n) for n in base_adjacency))
        else:
            log_dmax = 1.0
        self._degree_cache = (base_adjacency, log_dmax)
        return log_dmax

    @staticmethod
    def _bounded_distances(source, region, adjacency, max_depth):
        """Hop distances from ``source`` to region nodes within ``max_depth``.

        The BFS never leaves ``region``; nodes farther away (or unreachable)
        are simply absent and mapped to the cap by the feature builder.
        """
        distances = {source: 0}
        frontier = [source]
        for depth in range(1, max_depth + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor in region and neighbor not in distances:
                        distances[neighbor] = depth
                        next_frontier.append(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        return distances

    @staticmethod
    def _topo_features(candidates, u, v, selected, adjacency, dist_u, dist_v,
                       k_hops, log_dmax):
        """Deterministic topology features for one scoring step.

        Column layout matches ``STRUCTURAL_PRIOR``:

        .. code-block:: text

            0  common-neighbor indicator  1[c ∈ 𝒜_u ∩ 𝒜_v]
            1  target-touching count      𝟙[c ∈ 𝒜_u] + 𝟙[c ∈ 𝒜_v]
            2  normalized safe degree     log1p(|𝒜_c|) / log1p(d_max)
            3  selected-connectivity      |𝒜_c ∩ S| / |S|
            4  hop distance to u          min(d(c,u), 2·k_hops) / k_hops
            5  hop distance to v          min(d(c,v), 2·k_hops) / k_hops
            6  Adamic-Adar to u
            7  Adamic-Adar to v

        Every quantity is defined on the target-safe adjacency ``𝒜``.  The
        Adamic-Adar sums run over common neighbors ``w`` of the candidate and
        the respective target with weight ``1/log(2+deg(w))``; distances come
        from the per-trajectory bounded BFS, with missing entries (farther
        than the cap) taking the cap value.
        """
        cap = 2 * k_hops
        selected_size = len(selected)
        rows = []
        for candidate in candidates:
            neighbors = adjacency[candidate]
            in_u = candidate in adjacency[u]
            in_v = candidate in adjacency[v]
            rows.append([
                float(in_u and in_v),
                float(in_u) + float(in_v),
                math.log1p(len(neighbors)) / log_dmax,
                len(neighbors & selected) / selected_size,
                min(dist_u.get(candidate, cap), cap) / k_hops,
                min(dist_v.get(candidate, cap), cap) / k_hops,
                sum(
                    1.0 / math.log(2.0 + len(adjacency[w]))
                    for w in neighbors & adjacency[u]
                ),
                sum(
                    1.0 / math.log(2.0 + len(adjacency[w]))
                    for w in neighbors & adjacency[v]
                ),
            ])
        return torch.tensor(rows, dtype=torch.float32)

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

        Immutability lets one structure be built once per graph (the full
        dataset graph in the training entry) and shared by every target; each
        target excludes its own edge lazily in ``sample``.
        """
        adjacency = [set() for _ in range(num_nodes)]
        for source, target in edge_index.t().tolist():
            if source != target:
                adjacency[source].add(target)
                adjacency[target].add(source)
        return tuple(frozenset(neighbors) for neighbors in adjacency)

    @staticmethod
    def _k_hop_region(seed_nodes, adjacency, k_hops):
        """Return nodes within ``k_hops`` of the G0 seeds in safe adjacency.

        This local BFS runs once when a trajectory starts; action steps only
        maintain a frontier within the returned set.
        """
        allowed = set(seed_nodes)
        current_ring = set(seed_nodes)
        for _ in range(k_hops):
            next_ring = set()
            for node in current_ring:
                next_ring.update(adjacency[node])
            next_ring.difference_update(allowed)
            if not next_ring:
                break
            allowed.update(next_ring)
            current_ring = next_ring
        return allowed

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
    def _make_graph(selected, graph_edges, target_local, node_index,
                    edge_relations=None):
        device = node_index.device
        selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
        local = {global_node: i for i, global_node in enumerate(selected)}
        directed_edges = []
        ordered_edges = sorted(graph_edges)
        for source, target in ordered_edges:
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
        edge_attr = None
        if edge_relations is not None:
            edge_attr = edge_relations.lookup(ordered_edges).repeat_interleave(
                2, dim=0
            ).to(device)
        return SampledGraph(
            node_index[selected_tensor], selected_tensor, edge_index, target_nodes,
            edge_attr,
        )

    @staticmethod
    def _make_graphs(snapshots, target_local, node_index, edge_relations=None):
        """Materialize consecutive step graphs for one trajectory.

        Step snapshots contain monotonically growing ``selected`` node lists,
        so the local id map is extended incrementally across snapshots instead
        of being rebuilt from scratch per graph.  The produced tensors are
        identical to calling :meth:`_make_graph` on every snapshot.
        """
        device = node_index.device
        graphs = []
        local = {}
        for selected, graph_edges in snapshots:
            for node in selected[len(local):]:
                local[node] = len(local)
            selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
            directed_edges = []
            ordered_edges = sorted(graph_edges)
            for source, target in ordered_edges:
                directed_edges.extend(((local[source], local[target]),
                                       (local[target], local[source])))
            if directed_edges:
                edge_index = torch.tensor(
                    directed_edges, device=device, dtype=torch.long
                ).t()
            else:
                edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
            target_nodes = torch.tensor(
                [local[int(target_local[0])], local[int(target_local[1])]],
                device=device,
                dtype=torch.long,
            )
            edge_attr = None
            if edge_relations is not None:
                edge_attr = edge_relations.lookup(ordered_edges).repeat_interleave(
                    2, dim=0
                ).to(device)
            graphs.append(SampledGraph(
                node_index[selected_tensor], selected_tensor, edge_index,
                target_nodes, edge_attr,
            ))
        return graphs


class StaticNeighborhoodSampler(SubgraphSampler):
    """Non-learnable ablation sampler: G0 plus every safe k-hop neighbor.

    Unlike :class:`SubgraphSampler`, no nodes are chosen by a learned policy:
    the returned trajectory has no steps and its ``final_graph`` (equal to the
    baseline graph) is the induced safe subgraph on the whole ``k_hops`` region
    of the G0 seeds (``u``, ``v`` and any virtual proxies).  The trainer update
    phase is then a no-op (the trajectory carries no steps) and only the
    predictor is trained, exactly on the static neighborhood graph.

    The node set and edge rule match the candidate space the RL sampler is
    allowed to act in (the same ``_k_hop_region`` over the same safe
    adjacency), so comparing the two isolates the effect of learned selection:
    the static graph is a superset (information upper bound) of every RL
    trajectory graph for the same target.
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, max_steps=10, k_hops=1):
        # Deliberately no action-scoring parameters: this sampler is
        # non-learnable.  ``hidden_dim`` and ``max_steps`` are accepted only to
        # keep the constructor signature uniform with ``SubgraphSampler`` (the
        # values are not stored: this sampler never scores actions).
        if k_hops < 0:
            raise ValueError("k_hops must be non-negative")
        nn.Module.__init__(self)
        self.esm_dim = esm_dim
        self.k_hops = k_hops

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None, edge_relations=None):
        """Return one deterministic static-neighborhood trajectory.

        The ``training`` flag is ignored: the graph is always the full safe
        ``k_hops`` region of the G0 seeds, so sampling is deterministic.
        """
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )

        u, v = target_local.tolist()
        selected = [u, v]
        graph_edges = set()
        self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        allowed = self._k_hop_region(selected, adjacency, self.k_hops)
        # Take the whole region (seeds first, remaining safe k-hop nodes in a
        # deterministic order) and keep all safe edges induced by it.
        selected = selected + sorted(allowed - set(selected))
        self._add_real_edges(selected, graph_edges, adjacency)
        graph = self._make_graph(
            selected, graph_edges, target_local, node_index, edge_relations
        )
        return SamplingTrajectory(graph, [])


class RandomSubsetSampler(SubgraphSampler):
    """Non-learnable ablation sampler: a random subset of the k-hop region.

    Like :class:`StaticNeighborhoodSampler` this sampler has no parameters and
    returns a trajectory without steps, so the trainer only updates the
    predictor.  The node set is a uniformly random subset of the safe
    ``k_hops`` region with the same size budget as RL trajectories: the two
    target seeds are always included and ``min_size``..``max_size`` bound the
    final node count (when the region holds fewer candidates all of them are
    taken, mirroring early frontier exhaustion in RL).  Mandatory virtual
    proxies can push the final count above ``min_size`` (both targets
    isolated); it never exceeds ``max_size``.

    Comparing this sampler with the learned one at the same graph size
    separates the effect of the selection *strategy* from the effect of
    *context amount* (the static sampler is the full-region reference).
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, max_steps=10, k_hops=1,
                 min_size=3, max_size=7):
        # Deliberately no action-scoring parameters: this sampler is
        # non-learnable.  ``hidden_dim`` and ``max_steps`` are accepted only to
        # keep the constructor signature uniform with ``SubgraphSampler`` (the
        # values are not stored: this sampler never scores actions).
        if k_hops < 0:
            raise ValueError("k_hops must be non-negative")
        if min_size < 2:
            raise ValueError("min_size must be at least 2")
        if max_size < min_size:
            raise ValueError("max_size must be at least min_size")
        nn.Module.__init__(self)
        self.esm_dim = esm_dim
        self.k_hops = k_hops
        self.min_size = min_size
        self.max_size = max_size

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None, edge_relations=None):
        """Return one random-subset trajectory (deterministic under the seed).

        The ``training`` flag is ignored: the target size is drawn uniformly
        from ``[min_size, max_size]`` and the subset uniformly without
        replacement, both from the global RNG (seeded by the training entry),
        so results are reproducible per process seed.
        """
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )

        u, v = target_local.tolist()
        selected = [u, v]
        graph_edges = set()
        self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        allowed = self._k_hop_region(selected, adjacency, self.k_hops)
        candidates = sorted(allowed - set(selected))
        if candidates:
            target_size = int(torch.randint(self.min_size, self.max_size + 1, ()))
            # Mandatory virtual proxies can push ``len(selected)`` above
            # ``min_size`` (both targets isolated), so the extra budget is
            # clamped at zero; a negative slice would keep n-1 candidates
            # instead of none.
            k_extra = max(0, min(target_size - len(selected), len(candidates)))
            chosen = torch.randperm(len(candidates))[:k_extra].tolist()
            selected = selected + [candidates[index] for index in chosen]
        self._add_real_edges(selected, graph_edges, adjacency)
        graph = self._make_graph(
            selected, graph_edges, target_local, node_index, edge_relations
        )
        return SamplingTrajectory(graph, [])


class HeuristicSampler(SubgraphSampler):
    """Non-learnable ablation sampler: a hand-crafted same-budget selection.

    Zero parameters and no steps, like :class:`RandomSubsetSampler`, so the
    trainer only updates the predictor.  The node set is a deterministic,
    topology-only ranking of the safe ``k_hops`` region: common neighbors of
    both targets first, then nodes touching exactly one target, then the rest
    of the region; within each tier nodes are ranked by safe degree
    (descending) with node-id tie-breaks.  The size budget is identical to
    :class:`RandomSubsetSampler` (same uniform ``min_size``..``max_size``
    draw, mandatory seeds always included), so heuristic vs random vs static
    isolates the value of an informed selection rule at a fixed budget.
    """

    def __init__(self, esm_dim=2560, hidden_dim=512, max_steps=10, k_hops=1,
                 min_size=3, max_size=7):
        # Deliberately no action-scoring parameters: this sampler is
        # non-learnable.  ``hidden_dim`` and ``max_steps`` are accepted only to
        # keep the constructor signature uniform with ``SubgraphSampler`` (the
        # values are not stored: this sampler never scores actions).
        if k_hops < 0:
            raise ValueError("k_hops must be non-negative")
        if min_size < 2:
            raise ValueError("min_size must be at least 2")
        if max_size < min_size:
            raise ValueError("max_size must be at least min_size")
        nn.Module.__init__(self)
        self.esm_dim = esm_dim
        self.k_hops = k_hops
        self.min_size = min_size
        self.max_size = max_size

    def sample(self, node_features, edge_index, target_nodes, node_index=None,
               training=None, adjacency=None, edge_relations=None):
        """Return one heuristic trajectory (size drawn from the global RNG).

        The ``training`` flag is ignored: the ranking is deterministic, only
        the budget draw comes from the global RNG (seeded by the training
        entry), mirroring :class:`RandomSubsetSampler`.
        """
        node_features, edge_index, node_index, target_local = self._prepare_inputs(
            node_features, edge_index, target_nodes, node_index
        )
        adjacency = self._prepare_adjacency(
            edge_index, target_local, node_features.shape[0], adjacency
        )

        u, v = target_local.tolist()
        selected = [u, v]
        graph_edges = set()
        self._add_virtual_proxies(
            selected, graph_edges, adjacency, node_features, target_local
        )
        allowed = self._k_hop_region(selected, adjacency, self.k_hops)
        candidates = sorted(allowed - set(selected))
        if candidates:
            target_size = int(torch.randint(self.min_size, self.max_size + 1, ()))
            # Same clamp as RandomSubsetSampler: mandatory proxies may put
            # ``len(selected)`` above ``min_size``.
            k_extra = max(0, min(target_size - len(selected), len(candidates)))
            ranked = self._rank_candidates(candidates, u, v, adjacency)
            selected = selected + ranked[:k_extra]
        self._add_real_edges(selected, graph_edges, adjacency)
        graph = self._make_graph(
            selected, graph_edges, target_local, node_index, edge_relations
        )
        return SamplingTrajectory(graph, [])

    @staticmethod
    def _rank_candidates(candidates, u, v, adjacency):
        """Rank region nodes: common target neighbors first, then nodes
        touching exactly one target, then the rest; within each tier by
        descending safe degree with node-id tie-breaks."""
        def degree_key(node):
            return (-len(adjacency[node]), node)

        common = [c for c in candidates
                  if c in adjacency[u] and c in adjacency[v]]
        touching = [c for c in candidates
                    if (c in adjacency[u]) != (c in adjacency[v])]
        rest = [c for c in candidates
                if c not in adjacency[u] and c not in adjacency[v]]
        return (sorted(common, key=degree_key)
                + sorted(touching, key=degree_key)
                + sorted(rest, key=degree_key))
