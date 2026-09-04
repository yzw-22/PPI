import unittest

import torch

from src.sampler import (
    EdgeRelationLookup,
    RandomSubsetSampler,
    StaticNeighborhoodSampler,
    SubgraphSampler,
    _TargetSafeAdjacency,
)


def global_edges(graph):
    edges = set()
    for source, target in graph.edge_index.t().tolist():
        source = int(graph.node_index[source])
        target = int(graph.node_index[target])
        edges.add(tuple(sorted((source, target))))
    return edges


class EdgeRelationLookupTest(unittest.TestCase):
    def test_unknown_edges_are_zero_and_direction_does_not_matter(self):
        first = torch.tensor([1.0, 0.0, 1.0])
        second = torch.tensor([0.0, 1.0, 0.0])
        lookup = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 2], [1, 3]]),
            torch.stack((first, second)),
            num_nodes=4,
        )

        actual = lookup.lookup([[2, 0], [0, 1], [3, 1]])

        self.assertTrue(torch.equal(actual[0], first))
        self.assertTrue(torch.equal(actual[1], torch.zeros(3)))
        self.assertTrue(torch.equal(actual[2], second))

    def test_duplicate_undirected_edges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            EdgeRelationLookup.from_pairs(
                torch.tensor([[0, 1], [1, 0]]), torch.ones((2, 7)), num_nodes=2
            )


def _graph_signature(graph):
    return (
        graph.node_index.tolist(),
        tuple(sorted(tuple(sorted(edge)) for edge in graph.edge_index.t().tolist())),
        graph.target_nodes.tolist(),
    )


def _trajectory_signature(trajectory):
    return (
        _graph_signature(trajectory.baseline_graph),
        [_graph_signature(step.graph) for step in trajectory.steps],
    )


class SubgraphSamplerTest(unittest.TestCase):
    def test_step_graph_relations_align_with_directed_edges(self):
        node_features = torch.eye(3, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 2],
            [1, 0, 2, 0, 2, 1],
        ])
        target_relation = torch.tensor([1., 0., 0., 0., 0., 0., 0.])
        context_relation = torch.tensor([0., 0., 1., 0., 1., 0., 0.])
        relations = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 1], [0, 2]]),
            torch.stack((target_relation, context_relation)),
            num_nodes=3,
        )
        sampler = SubgraphSampler(esm_dim=3, hidden_dim=2, max_steps=1)

        trajectory = sampler.sample(
            node_features,
            edge_index,
            torch.tensor([0, 1]),
            training=False,
            edge_relations=relations,
        )

        self.assertEqual(trajectory.baseline_graph.edge_attr.shape, (0, 7))
        self.assertTrue(torch.equal(
            trajectory.final_graph.edge_attr,
            torch.stack((
                context_relation,
                context_relation,
                torch.zeros(7),
                torch.zeros(7),
            )),
        ))

    def test_action_score_uses_residual_pair_projection(self):
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=4, max_steps=1)

        self.assertEqual(sampler.pair_proj.in_features, 8)
        self.assertEqual(sampler.pair_proj.out_features, 4)
        self.assertEqual(sampler.fc[0].in_features, 4)
        self.assertEqual(sampler.fc[-1].out_features, 1)
        self.assertFalse(hasattr(sampler, "fixed_num"))
        self.assertEqual(sampler.k_hops, 1)

    def test_candidate_relations_are_or_aggregated_over_selected_edges(self):
        sampler = SubgraphSampler(
            esm_dim=2, hidden_dim=4, max_steps=1, relation_dim=3
        )
        adjacency = sampler._build_adjacency(
            torch.tensor([
                [0, 2, 1, 2, 1, 3],
                [2, 0, 2, 1, 3, 1],
            ]),
            5,
        )
        relations = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 2], [1, 2], [1, 3]]),
            torch.tensor([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]),
            num_nodes=5,
        )

        actual = sampler._candidate_relation_features(
            [2, 3, 4], [0, 1], adjacency, relations
        )

        self.assertTrue(torch.equal(actual, torch.tensor([
            [1., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])))

    def test_relation_features_change_greedy_action_without_target_leakage(self):
        node_features = torch.zeros((4, 2))
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 1, 2],
            [1, 0, 2, 0, 3, 0, 2, 1],
        ])
        sampler = SubgraphSampler(
            esm_dim=2, hidden_dim=4, max_steps=1, relation_dim=2
        )
        with torch.no_grad():
            sampler.state_proj.weight.zero_()
            sampler.neighbor_proj.weight.zero_()
            sampler.relation_proj.weight.zero_()
            sampler.relation_proj.weight[0, 0] = 1.0
            sampler.pair_proj.weight.zero_()
            sampler.pair_proj.bias.zero_()
            sampler.pair_proj.weight[0, 4] = 1.0
            sampler.fc[0].weight.zero_()
            sampler.fc[0].bias.zero_()
            sampler.fc[0].weight[0, 0] = 1.0
            sampler.fc[0].weight[1, 0] = -1.0
            sampler.fc[3].weight.copy_(torch.tensor([[1.0, -1.0]]))
            sampler.fc[3].bias.zero_()

        # The target relation is present in both lookups, but the safe
        # adjacency removes 0-1 before policy features are constructed.
        target_relation = torch.tensor([0., 1.])
        relation_on_two = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 1], [0, 2]]),
            torch.stack((target_relation, torch.tensor([1., 0.]))),
            num_nodes=4,
        )
        relation_on_three = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 1], [0, 3]]),
            torch.stack((target_relation, torch.tensor([1., 0.]))),
            num_nodes=4,
        )

        first = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False,
            edge_relations=relation_on_two,
        )
        second = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False,
            edge_relations=relation_on_three,
        )

        self.assertEqual(first.final_graph.node_index.tolist()[-1], 2)
        self.assertEqual(second.final_graph.node_index.tolist()[-1], 3)

    def test_relation_projection_receives_policy_gradient(self):
        torch.manual_seed(7)
        node_features = torch.zeros((4, 2))
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 1, 2],
            [1, 0, 2, 0, 3, 0, 2, 1],
        ])
        relations = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 2], [0, 3]]),
            torch.tensor([[1., 0.], [0., 1.]]),
            num_nodes=4,
        )
        sampler = SubgraphSampler(
            esm_dim=2, hidden_dim=4, max_steps=1, relation_dim=2
        )

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=True,
            edge_relations=relations,
        )
        (-trajectory.steps[0].log_prob).backward()

        gradient = sampler.relation_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_relation_branch_preserves_legacy_rng_and_common_parameters(self):
        torch.manual_seed(19)
        plain = SubgraphSampler(esm_dim=2, hidden_dim=4, max_steps=1)
        after_plain = torch.rand(4)
        torch.manual_seed(19)
        relation_aware = SubgraphSampler(
            esm_dim=2, hidden_dim=4, max_steps=1, relation_dim=7
        )
        after_relation = torch.rand(4)

        relation_state = relation_aware.state_dict()
        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, relation_state[name]))
        self.assertTrue(torch.equal(after_plain, after_relation))

    def test_relation_aware_sampler_validates_lookup(self):
        sampler = SubgraphSampler(
            esm_dim=2, hidden_dim=4, max_steps=1, relation_dim=2
        )
        node_features = torch.zeros((3, 2))
        edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])

        with self.assertRaisesRegex(ValueError, "required"):
            sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False
            )
        wrong_dim = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 2]]), torch.ones((1, 3)), num_nodes=3
        )
        with self.assertRaisesRegex(ValueError, "dimension 2"):
            sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False,
                edge_relations=wrong_dim,
            )

    def test_negative_k_hops_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "k_hops"):
            SubgraphSampler(esm_dim=2, k_hops=-1)

    def test_node_index_must_be_strictly_increasing(self):
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=2, max_steps=0)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sampler.sample(
                torch.zeros((3, 2)),
                torch.empty((2, 0), dtype=torch.long),
                torch.tensor([10, 20]),
                node_index=torch.tensor([20, 10, 30]),
                training=False,
            )

    def test_baseline_graph_contains_induced_real_edges(self):
        # Target edge 0-1 is removed; safe edge 1-2 remains available to the
        # frontier but is not included in G0.
        node_features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 1, 0],
            [1, 0, 2, 2],
        ])
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=2, max_steps=0)

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(set(trajectory.baseline_graph.node_index.tolist()), {0, 1})
        self.assertEqual(global_edges(trajectory.baseline_graph), set())
        self.assertNotIn((0, 1), global_edges(trajectory.baseline_graph))

    def test_isolated_targets_can_share_one_proxy_without_duplicates(self):
        # Both targets prefer node 2. Target nodes must never be proxy candidates.
        node_features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=2, max_steps=0)

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(trajectory.baseline_graph.node_index.tolist(), [0, 1, 2])
        self.assertEqual(global_edges(trajectory.baseline_graph), {(0, 2), (1, 2)})
        self.assertNotIn(0, trajectory.baseline_graph.node_index[2:].tolist())
        self.assertNotIn(1, trajectory.baseline_graph.node_index[2:].tolist())
        self.assertNotIn((0, 1), global_edges(trajectory.baseline_graph))

    def test_baseline_graph_excludes_initial_neighbors_but_frontier_expands(self):
        node_features = torch.eye(6, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 3, 3, 1, 3, 0, 2, 5, 3, 4],
        ])
        sampler = SubgraphSampler(esm_dim=6, hidden_dim=2, max_steps=0)

        first = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )
        self.assertEqual(first.baseline_graph.node_index.tolist(), [0, 1])
        self.assertNotIn((0, 1), global_edges(first.baseline_graph))

        sampler.max_steps = 1
        expanded = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )
        self.assertEqual(len(expanded.steps), 1)
        self.assertEqual(len(expanded.final_graph.node_index), 3)
        self.assertTrue(
            set(expanded.final_graph.node_index.tolist()).intersection({2, 3})
        )

    def test_k_hop_region_and_trajectory_limit(self):
        # Target edge 0-1 is removed. Nodes 2, 3, and 4 are respectively one,
        # two, and three safe hops away from the G0 target pair.
        node_features = torch.eye(5, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2, 3],
            [1, 0, 2, 2, 3, 4],
        ])
        adjacency = SubgraphSampler._build_adjacency(edge_index, 5)
        self.assertEqual(
            SubgraphSampler._k_hop_region([0, 1], adjacency, 0), {0, 1}
        )
        self.assertEqual(
            SubgraphSampler._k_hop_region([0, 1], adjacency, 1), {0, 1, 2}
        )
        self.assertEqual(
            SubgraphSampler._k_hop_region([0, 1], adjacency, 2), {0, 1, 2, 3}
        )

        for k_hops, expected_nodes in ((0, {0, 1}), (1, {0, 1, 2}),
                                       (2, {0, 1, 2, 3})):
            sampler = SubgraphSampler(
                esm_dim=5, hidden_dim=2, max_steps=10, k_hops=k_hops
            )
            trajectory = sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False
            )
            self.assertEqual(
                set(trajectory.final_graph.node_index.tolist()), expected_nodes
            )
            self.assertNotIn((0, 1), global_edges(trajectory.final_graph))

    def test_k_hop_region_is_built_once_and_shared_adjacency_is_lazily_overlaid(self):
        class CountingSampler(SubgraphSampler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.k_hop_region_calls = 0

            def _k_hop_region(self, *args):
                self.k_hop_region_calls += 1
                return super()._k_hop_region(*args)

        node_features = torch.eye(4, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2],
            [1, 0, 2, 2, 3],
        ])
        shared = SubgraphSampler._build_adjacency(edge_index, 4)
        sampler = CountingSampler(esm_dim=4, hidden_dim=2, max_steps=3, k_hops=2)
        safe_adjacency = sampler._prepare_adjacency(
            edge_index, torch.tensor([0, 1]), 4, shared
        )

        self.assertIsInstance(safe_adjacency, _TargetSafeAdjacency)
        self.assertIs(safe_adjacency.base, shared)
        self.assertNotIn(1, safe_adjacency[0])
        self.assertNotIn(0, safe_adjacency[1])

        sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False,
            adjacency=shared,
        )
        self.assertEqual(sampler.k_hop_region_calls, 1)

    def test_baseline_graph_has_no_initial_context_edges(self):
        node_features = torch.eye(4, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],
            [1, 0, 2, 0, 2, 1, 3, 1, 3, 2],
        ])
        sampler = SubgraphSampler(esm_dim=4, hidden_dim=2, max_steps=0)
        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(set(trajectory.baseline_graph.node_index.tolist()), {0, 1})
        self.assertEqual(global_edges(trajectory.baseline_graph), set())

    def test_shared_adjacency_is_equivalent_to_standalone_build(self):
        # A shared immutable adjacency (with the target edge still present),
        # patched lazily per target, must produce the same trajectory as the
        # standalone path that builds a safe adjacency with the edge removed.
        node_features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
        ])
        target = torch.tensor([0, 1])
        shared = SubgraphSampler._build_adjacency(edge_index, 5)
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=4, max_steps=2)

        standalone = sampler.sample(node_features, edge_index, target, training=False)
        shared_trajectory = sampler.sample(
            node_features, edge_index, target, training=False, adjacency=shared
        )

        self.assertEqual(
            _trajectory_signature(shared_trajectory),
            _trajectory_signature(standalone),
        )


class StaticNeighborhoodSamplerTest(unittest.TestCase):
    def test_only_visible_non_target_edges_receive_relations(self):
        # 0-1 is the target and must disappear. Relation visibility contains
        # target 0-1 and train edge 0-2, but not held-out topology edge 1-3.
        node_features = torch.eye(4, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 3],
            [1, 0, 2, 0, 3, 1],
        ])
        target_relation = torch.tensor([1., 0., 0., 0., 0., 0., 0.])
        train_relation = torch.tensor([0., 1., 1., 0., 0., 0., 0.])
        relations = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 1], [0, 2]]),
            torch.stack((target_relation, train_relation)),
            num_nodes=4,
        )
        sampler = StaticNeighborhoodSampler(esm_dim=4, k_hops=1)

        graph = sampler.sample(
            node_features,
            edge_index,
            torch.tensor([0, 1]),
            training=False,
            edge_relations=relations,
        ).final_graph

        attributed_edges = {}
        for edge, relation in zip(graph.edge_index.t(), graph.edge_attr):
            source = int(graph.node_index[int(edge[0])])
            target = int(graph.node_index[int(edge[1])])
            attributed_edges[(source, target)] = relation
        self.assertNotIn((0, 1), attributed_edges)
        self.assertNotIn((1, 0), attributed_edges)
        self.assertTrue(torch.equal(attributed_edges[(0, 2)], train_relation))
        self.assertTrue(torch.equal(attributed_edges[(2, 0)], train_relation))
        self.assertTrue(torch.equal(attributed_edges[(1, 3)], torch.zeros(7)))
        self.assertTrue(torch.equal(attributed_edges[(3, 1)], torch.zeros(7)))

    def test_virtual_proxy_edges_receive_zero_relations(self):
        node_features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        relations = EdgeRelationLookup.from_pairs(
            torch.tensor([[0, 1]]), torch.ones((1, 7)), num_nodes=3
        )
        sampler = StaticNeighborhoodSampler(esm_dim=2, k_hops=1)

        graph = sampler.sample(
            node_features,
            edge_index,
            torch.tensor([0, 1]),
            training=False,
            edge_relations=relations,
        ).final_graph

        self.assertTrue(torch.equal(graph.edge_attr, torch.zeros_like(graph.edge_attr)))

    def test_has_no_learnable_parameters_and_validates_k_hops(self):
        sampler = StaticNeighborhoodSampler(esm_dim=2, hidden_dim=4, k_hops=1)

        self.assertEqual(list(sampler.parameters()), [])
        self.assertEqual(sampler.k_hops, 1)
        with self.assertRaisesRegex(ValueError, "k_hops"):
            StaticNeighborhoodSampler(esm_dim=2, k_hops=-1)

    def test_trajectory_has_no_steps_and_final_is_the_baseline_graph(self):
        node_features = torch.eye(5, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2, 3],
            [1, 0, 2, 2, 3, 4],
        ])
        sampler = StaticNeighborhoodSampler(esm_dim=5, k_hops=1)

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(trajectory.steps, [])
        self.assertIs(trajectory.final_graph, trajectory.baseline_graph)

    def test_graph_is_the_full_k_hop_region_with_induced_safe_edges(self):
        # Undirected edges: 0-1 (target, removed), 0-2, 1-2, 2-3, 3-4.
        # k=1 region of {0,1} is {0,1,2}; k=2 region is {0,1,2,3}.
        node_features = torch.eye(5, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2, 3],
            [1, 0, 2, 2, 3, 4],
        ])
        for k_hops, expected_nodes, expected_edges in (
            (1, {0, 1, 2}, {(0, 2), (1, 2)}),
            (2, {0, 1, 2, 3}, {(0, 2), (1, 2), (2, 3)}),
        ):
            sampler = StaticNeighborhoodSampler(esm_dim=5, k_hops=k_hops)
            graph = sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False
            ).final_graph

            self.assertEqual(set(graph.node_index.tolist()), expected_nodes)
            self.assertEqual(global_edges(graph), expected_edges)
            self.assertNotIn((0, 1), global_edges(graph))

    def test_isolated_target_takes_proxy_and_its_one_hop_region(self):
        # The only edge 0-1 is the target; both targets prefer node 2 as proxy.
        node_features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        sampler = StaticNeighborhoodSampler(esm_dim=2, k_hops=1)

        graph = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        ).final_graph

        self.assertEqual(graph.node_index.tolist(), [0, 1, 2])
        self.assertEqual(global_edges(graph), {(0, 2), (1, 2)})
        self.assertNotIn((0, 1), global_edges(graph))

    def test_training_flag_does_not_change_the_static_graph(self):
        node_features = torch.eye(4, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 2, 1, 3],
            [1, 0, 2, 0, 2, 1, 3, 1],
        ])
        sampler = StaticNeighborhoodSampler(esm_dim=4, k_hops=1)

        train_trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=True
        )
        eval_trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(
            _trajectory_signature(train_trajectory),
            _trajectory_signature(eval_trajectory),
        )

    def test_shared_adjacency_is_equivalent_to_standalone_build(self):
        node_features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
        ])
        target = torch.tensor([0, 1])
        shared = SubgraphSampler._build_adjacency(edge_index, 5)
        sampler = StaticNeighborhoodSampler(esm_dim=2, k_hops=1)

        standalone = sampler.sample(node_features, edge_index, target, training=False)
        shared_trajectory = sampler.sample(
            node_features, edge_index, target, training=False, adjacency=shared
        )

        self.assertEqual(
            _trajectory_signature(shared_trajectory),
            _trajectory_signature(standalone),
        )


class RandomSubsetSamplerTest(unittest.TestCase):
    def test_has_no_learnable_parameters_and_validates_size_range(self):
        sampler = RandomSubsetSampler(esm_dim=2, k_hops=1, min_size=3, max_size=5)

        self.assertEqual(list(sampler.parameters()), [])
        self.assertEqual((sampler.min_size, sampler.max_size), (3, 5))
        with self.assertRaisesRegex(ValueError, "k_hops"):
            RandomSubsetSampler(esm_dim=2, k_hops=-1)
        with self.assertRaisesRegex(ValueError, "min_size"):
            RandomSubsetSampler(esm_dim=2, min_size=1)
        with self.assertRaisesRegex(ValueError, "max_size"):
            RandomSubsetSampler(esm_dim=2, min_size=5, max_size=3)

    def test_trajectory_has_no_steps_and_final_is_the_baseline_graph(self):
        node_features = torch.eye(8, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 1, 7],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 6, 1, 7, 1],
        ])
        sampler = RandomSubsetSampler(esm_dim=8, k_hops=1, min_size=3, max_size=5)

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(trajectory.steps, [])
        self.assertIs(trajectory.final_graph, trajectory.baseline_graph)

    def test_graph_always_contains_targets_within_the_region_and_size_range(self):
        # Region of {0, 1} is {0..7}: candidates 2..7 give sizes 3..5 room.
        node_features = torch.eye(8, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 1, 7],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 6, 1, 7, 1],
        ])
        sampler = RandomSubsetSampler(esm_dim=8, k_hops=1, min_size=3, max_size=5)

        sizes = set()
        for seed in range(100):
            torch.manual_seed(seed)
            graph = sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False
            ).final_graph
            nodes = set(graph.node_index.tolist())
            self.assertLessEqual(nodes, {0, 1, 2, 3, 4, 5, 6, 7})
            self.assertIn(0, nodes)
            self.assertIn(1, nodes)
            self.assertNotIn((0, 1), global_edges(graph))
            sizes.add(len(nodes))
        self.assertTrue(sizes <= {3, 4, 5})
        self.assertGreater(len(sizes), 1)  # the size draw is not constant

    def test_edges_are_the_induced_safe_edges_of_the_drawn_nodes(self):
        # Safe edges: 0-2, 0-3, 0-4, 1-5, 1-6, 1-7 (0-1 is the target edge).
        node_features = torch.eye(8, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 1, 7],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 6, 1, 7, 1],
        ])
        safe_edges = {(0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)}
        sampler = RandomSubsetSampler(esm_dim=8, k_hops=1, min_size=3, max_size=5)

        torch.manual_seed(7)
        graph = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        ).final_graph
        nodes = set(graph.node_index.tolist())
        expected = {tuple(sorted(edge)) for edge in safe_edges
                    if set(edge) <= nodes}
        self.assertEqual(global_edges(graph), expected)

    def test_exhausted_region_takes_all_candidates(self):
        # Region of {0, 1} is {0, 1, 2}; sizes above the region size clamp to
        # all candidates, like early frontier exhaustion in RL.
        node_features = torch.eye(5, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2, 3],
            [1, 0, 2, 2, 3, 4],
        ])
        sampler = RandomSubsetSampler(esm_dim=5, k_hops=1, min_size=3, max_size=5)

        for seed in range(10):
            torch.manual_seed(seed)
            graph = sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False
            ).final_graph

            self.assertEqual(set(graph.node_index.tolist()), {0, 1, 2})
            self.assertEqual(global_edges(graph), {(0, 2), (1, 2)})
            self.assertNotIn((0, 1), global_edges(graph))

    def test_training_flag_does_not_change_the_graph_under_the_same_seed(self):
        node_features = torch.eye(8, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 1, 7],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 6, 1, 7, 1],
        ])
        sampler = RandomSubsetSampler(esm_dim=8, k_hops=1, min_size=3, max_size=5)

        torch.manual_seed(3)
        train_trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=True
        )
        torch.manual_seed(3)
        eval_trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(
            _trajectory_signature(train_trajectory),
            _trajectory_signature(eval_trajectory),
        )

    def test_shared_adjacency_is_equivalent_to_standalone_build(self):
        node_features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
        ])
        target = torch.tensor([0, 1])
        shared = SubgraphSampler._build_adjacency(edge_index, 5)
        sampler = RandomSubsetSampler(esm_dim=2, k_hops=1, min_size=3, max_size=4)

        torch.manual_seed(11)
        standalone = sampler.sample(node_features, edge_index, target, training=False)
        torch.manual_seed(11)
        shared_trajectory = sampler.sample(
            node_features, edge_index, target, training=False, adjacency=shared
        )

        self.assertEqual(
            _trajectory_signature(shared_trajectory),
            _trajectory_signature(standalone),
        )


class RandomSubsetSamplerTwoIsolatedTargetsTest(unittest.TestCase):
    def test_two_isolated_targets_never_exceed_max_size(self):
        # Both targets are isolated, so each takes a distinct proxy and
        # len(selected) = 4 exceeds min_size = 3. When the drawn target size
        # is 3 the extra budget must clamp to zero; a negative slice would
        # keep n-1 candidates and blow past max_size.
        node_features = torch.zeros(10, 2)
        node_features[0] = torch.tensor([1.0, 0.3])
        node_features[2] = torch.tensor([1.0, 0.0])   # proxy for target 0
        node_features[1] = torch.tensor([-1.0, 0.3])
        node_features[3] = torch.tensor([-1.0, 0.0])  # proxy for target 1
        edge_index = torch.tensor([
            [0, 1, 2, 4, 2, 5, 2, 6, 3, 7, 3, 8, 3, 9],
            [1, 0, 4, 2, 5, 2, 6, 2, 7, 3, 8, 3, 9, 3],
        ])
        sampler = RandomSubsetSampler(esm_dim=2, k_hops=1, min_size=3, max_size=7)

        sizes = set()
        for seed in range(50):
            torch.manual_seed(seed)
            graph = sampler.sample(
                node_features, edge_index, torch.tensor([0, 1]), training=False
            ).final_graph
            nodes = set(graph.node_index.tolist())
            self.assertIn(0, nodes)
            self.assertIn(1, nodes)
            self.assertIn(2, nodes)  # both proxies are mandatory seeds
            self.assertIn(3, nodes)
            self.assertNotIn((0, 1), global_edges(graph))
            sizes.add(len(nodes))
        self.assertTrue(max(sizes) <= 7)
        self.assertTrue(max(sizes) > 4)  # a size-3 draw must still add nodes


if __name__ == "__main__":
    unittest.main()
