import unittest

import torch

from src.sampler import (
    RandomIterativeSubgraphSampler,
    RandomSubgraphSampler,
    SubgraphSampler,
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
    def test_max_steps_limits_learned_actions(self):
        node_features = torch.eye(8, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7],
            [1, 0, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4],
        ])
        sampler = SubgraphSampler(
            esm_dim=8, hidden_dim=4, max_steps=2, fixed_num=1,
        )
        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )
        self.assertLessEqual(len(trajectory.steps), sampler.max_steps)

    def test_action_score_uses_pairwise_mlp(self):
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=4, max_steps=1)

        self.assertEqual(sampler.action_mlp[0].in_features, 8)
        self.assertEqual(sampler.action_mlp[-1].out_features, 1)
        self.assertEqual(sampler.fixed_num, 1)

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
        # Target edge 0-1 is removed; 1-2 remains a real safe edge.
        node_features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 1],
            [1, 0, 2],
        ])
        sampler = SubgraphSampler(esm_dim=2, hidden_dim=2, max_steps=0)

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(set(trajectory.baseline_graph.node_index.tolist()), {0, 1, 2})
        self.assertEqual(global_edges(trajectory.baseline_graph), {(0, 2), (1, 2)})
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

    def test_baseline_graph_caps_one_hop_context_and_is_reproducible(self):
        node_features = torch.eye(6, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 3, 3, 1, 3, 0, 2, 5, 3, 4],
        ])
        sampler = SubgraphSampler(
            esm_dim=6, hidden_dim=2, max_steps=0, fixed_num=2
        )

        first = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )
        second = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(
            first.baseline_graph.node_index.tolist(),
            second.baseline_graph.node_index.tolist(),
        )
        self.assertNotIn((0, 1), global_edges(first.baseline_graph))

        initial_nodes = {0, 1}
        baseline_nodes = set(first.baseline_graph.node_index.tolist())
        for seed in initial_nodes:
            sampled_from_seed = baseline_nodes.intersection(
                set(SubgraphSampler._build_adjacency(edge_index, 6)[seed])
            ) - initial_nodes
            self.assertLessEqual(len(sampled_from_seed), sampler.fixed_num)

    def test_baseline_graph_contains_all_induced_safe_edges(self):
        node_features = torch.eye(4, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],
            [1, 0, 2, 0, 2, 1, 3, 1, 3, 2],
        ])
        sampler = SubgraphSampler(
            esm_dim=4, hidden_dim=2, max_steps=0, fixed_num=10
        )
        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(set(trajectory.baseline_graph.node_index.tolist()), {0, 1, 2, 3})
        self.assertEqual(
            global_edges(trajectory.baseline_graph),
            {(0, 2), (1, 2), (1, 3), (2, 3)},
        )

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


class RandomSubgraphSamplerTest(unittest.TestCase):
    def test_random_target_only_has_no_proxy_or_context(self):
        node_features = torch.eye(4, dtype=torch.float32)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        sampler = RandomSubgraphSampler(
            esm_dim=4, max_context_nodes=10, use_proxy=False
        )
        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )
        self.assertEqual(trajectory.steps, [])
        self.assertEqual(trajectory.proxy_nodes, ())
        self.assertEqual(trajectory.final_graph.node_index.tolist(), [0, 1])

    def test_random_one_hop_graph_is_reproducible_and_bounded(self):
        node_features = torch.eye(6, dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 0, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 3, 3, 1, 3, 0, 2, 1],
        ])
        sampler = RandomSubgraphSampler(
            esm_dim=6, max_context_nodes=2, random_seed=42
        )
        target = torch.tensor([0, 1])

        first = sampler.sample(node_features, edge_index, target, training=False)
        second = sampler.sample(node_features, edge_index, target, training=True)

        self.assertEqual(_trajectory_signature(first), _trajectory_signature(second))
        self.assertEqual(first.steps, [])
        self.assertLessEqual(len(first.final_graph.node_index), 4)
        self.assertNotIn((0, 1), global_edges(first.final_graph))
        safe_adjacency = SubgraphSampler._build_adjacency(edge_index, 6)
        one_hop = (set(safe_adjacency[0]) | set(safe_adjacency[1])) - {0, 1}
        context = set(first.final_graph.node_index.tolist()) - {0, 1}
        self.assertTrue(context.issubset(one_hop))

    def test_random_sampler_can_add_proxy_for_isolated_target(self):
        node_features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        sampler = RandomSubgraphSampler(esm_dim=2, max_context_nodes=2)

        trajectory = sampler.sample(
            node_features, edge_index, torch.tensor([0, 1]), training=False
        )

        self.assertEqual(trajectory.steps, [])
        self.assertIn(2, trajectory.final_graph.node_index.tolist())
        self.assertNotIn((0, 1), global_edges(trajectory.final_graph))
        self.assertEqual(list(sampler.parameters()), [])


class RandomIterativeSubgraphSamplerTest(unittest.TestCase):
    def setUp(self):
        self.node_features = torch.eye(14, dtype=torch.float32)
        # A chain plus branches gives the frontier multiple choices at each
        # step, while retaining a safe target edge to remove.
        self.edge_index = torch.tensor([
            [0, 1, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7,
             6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13],
            [1, 0, 2, 1, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5,
             8, 6, 9, 7, 10, 8, 11, 9, 12, 10, 13, 11],
        ])

    def test_random_iterative_is_reproducible_and_unparameterized(self):
        sampler = RandomIterativeSubgraphSampler(
            esm_dim=14, fixed_num=1, max_steps=10, random_seed=42
        )
        target = torch.tensor([0, 1])
        first = sampler.sample(self.node_features, self.edge_index, target)
        second = sampler.sample(self.node_features, self.edge_index, target)

        self.assertEqual(_trajectory_signature(first), _trajectory_signature(second))
        self.assertEqual(list(sampler.parameters()), [])
        self.assertLessEqual(len(first.steps), 10)

    def test_random_iterative_adds_one_frontier_node_per_step(self):
        sampler = RandomIterativeSubgraphSampler(
            esm_dim=14, fixed_num=0, max_steps=10, random_seed=42
        )
        trajectory = sampler.sample(
            self.node_features, self.edge_index, torch.tensor([0, 1])
        )
        previous = len(trajectory.baseline_graph.node_index)
        for step in trajectory.steps:
            current = len(step.graph.node_index)
            self.assertEqual(current - previous, 1)
            previous = current
        self.assertLessEqual(
            len(trajectory.final_graph.node_index)
            - len(trajectory.baseline_graph.node_index),
            10,
        )
        self.assertNotIn((0, 1), global_edges(trajectory.final_graph))


if __name__ == "__main__":
    unittest.main()
