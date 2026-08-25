import unittest

import torch

from src.sampler import (
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
    def test_action_score_uses_residual_pair_projection(self):
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=4, max_steps=1)

        self.assertEqual(sampler.pair_proj.in_features, 8)
        self.assertEqual(sampler.pair_proj.out_features, 4)
        self.assertEqual(sampler.fc[0].in_features, 4)
        self.assertEqual(sampler.fc[-1].out_features, 1)
        self.assertFalse(hasattr(sampler, "fixed_num"))
        self.assertEqual(sampler.k_hops, 1)

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


if __name__ == "__main__":
    unittest.main()
