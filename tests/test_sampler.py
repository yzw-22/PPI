import unittest

import torch

from src.sampler import SubgraphSampler


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
        self.assertLessEqual(
            len(first.baseline_graph.node_index), 2 + 2 * sampler.fixed_num
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


if __name__ == "__main__":
    unittest.main()
