import unittest

import torch

from src.ppr import PPRLookup


def _exact_ppr(edge_index, num_nodes, seed, alpha, iterations=2000):
    """Dense power-iteration reference for tests."""
    adj = torch.zeros(num_nodes, num_nodes)
    for source, target in edge_index.t().tolist():
        adj[source, target] = 1.0
        adj[target, source] = 1.0
    degree = adj.sum(dim=1)
    transition = torch.where(
        degree.unsqueeze(1) > 0, adj / degree.unsqueeze(1).clamp_min(1),
        torch.full_like(adj, 1.0 / num_nodes),
    )
    p = torch.zeros(num_nodes)
    p[seed] = 1.0
    for _ in range(iterations):
        p = alpha * torch.eye(num_nodes)[seed] + (1 - alpha) * (transition.t() @ p)
    return {node: float(p[node]) for node in range(num_nodes)}


class PPRLookupTest(unittest.TestCase):
    def test_rows_match_power_iteration_reference(self):
        # Undirected edges: 0-1, 0-2, 1-2, 2-3.
        edge_index = torch.tensor([
            [0, 1, 1, 0, 2, 2, 2, 3],
            [1, 0, 2, 2, 1, 0, 3, 2],
        ])
        lookup = PPRLookup(edge_index, num_nodes=4, alpha=0.15, eps=1e-10)

        for seed in range(4):
            row = lookup.get(seed)
            reference = _exact_ppr(edge_index, 4, seed, alpha=0.15)
            for node, value in reference.items():
                self.assertAlmostEqual(row.get(node, 0.0), value, places=5)

    def test_truncation_keeps_the_row_sparse(self):
        edge_index = torch.tensor([
            [0, 1, 1, 0, 2, 2, 2, 3],
            [1, 0, 2, 2, 1, 0, 3, 2],
        ])
        exact = PPRLookup(edge_index, num_nodes=4, alpha=0.15, eps=1e-10).get(0)
        coarse = PPRLookup(edge_index, num_nodes=4, alpha=0.15, eps=0.2).get(0)

        # With eps=0.2 the walk never reaches node 3 (distance 2, degree 1).
        self.assertIn(3, exact)
        self.assertNotIn(3, coarse)
        self.assertLessEqual(sum(coarse.values()), 1.0 + 1e-6)

    def test_rows_are_cached(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        lookup = PPRLookup(edge_index, num_nodes=3, alpha=0.15, eps=1e-8)

        first = lookup.get(0)
        second = lookup.get(0)
        self.assertIs(first, second)  # cache hit, no recomputation

        with self.assertRaisesRegex(ValueError, "out of range"):
            lookup.get(3)


if __name__ == "__main__":
    unittest.main()
