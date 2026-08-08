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


class SubgraphSamplerTest(unittest.TestCase):
    def test_initial_graph_contains_induced_real_edges(self):
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

        self.assertEqual(set(trajectory.initial_graph.node_index.tolist()), {0, 1, 2})
        self.assertEqual(global_edges(trajectory.initial_graph), {(0, 2), (1, 2)})
        self.assertNotIn((0, 1), global_edges(trajectory.initial_graph))

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

        self.assertEqual(trajectory.initial_graph.node_index.tolist(), [0, 1, 2])
        self.assertEqual(global_edges(trajectory.initial_graph), {(0, 2), (1, 2)})
        self.assertNotIn(0, trajectory.initial_graph.node_index[2:].tolist())
        self.assertNotIn(1, trajectory.initial_graph.node_index[2:].tolist())
        self.assertNotIn((0, 1), global_edges(trajectory.initial_graph))


if __name__ == "__main__":
    unittest.main()
