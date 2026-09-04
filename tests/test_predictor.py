import unittest

import torch

from src.ppr import PPRLookup
from src.predictor import PPIPredictor


class PPIPredictorTest(unittest.TestCase):
    def test_relation_features_flow_through_gat(self):
        predictor = PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, dropout=0.0,
            edge_dim=7,
        )
        features = torch.randn(2, 2)
        edges = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([
            [1., 0., 1., 0., 0., 0., 0.],
            [1., 0., 1., 0., 0., 0., 0.],
        ])

        logits = predictor(features, edges, torch.tensor([0, 1]), edge_attr=edge_attr)
        logits.sum().backward()

        self.assertEqual(logits.shape, (7,))
        self.assertIsNotNone(predictor.convs[0].lin_edge.weight.grad)
        self.assertTrue(torch.isfinite(predictor.convs[0].lin_edge.weight.grad).all())

    def test_relation_mode_validates_edge_attributes(self):
        predictor = PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, edge_dim=7
        )
        features = torch.randn(2, 2)
        edges = torch.tensor([[0, 1], [1, 0]])

        with self.assertRaisesRegex(ValueError, "required"):
            predictor(features, edges, torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "shape"):
            predictor(
                features, edges, torch.tensor([0, 1]), edge_attr=torch.ones((2, 6))
            )

class AttentionReadoutTest(unittest.TestCase):
    def _predictor(self, **kwargs):
        # Graph ids 5-7 and 11-13; the lookup must cover every global id.
        edge_index = torch.tensor([[5, 7], [7, 5]])
        return PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, dropout=0.0,
            ppr=PPRLookup(edge_index, num_nodes=14, alpha=0.15, eps=1e-8),
            **kwargs,
        )

    def test_flag_off_keeps_the_historical_architecture(self):
        predictor = PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, dropout=0.0
        )

        self.assertEqual(predictor.readout, "mean")
        self.assertIsNone(predictor.ppr)
        self.assertFalse(hasattr(predictor, "link_attention"))
        self.assertEqual(predictor.output[0].in_features, 12)  # 3 * hidden

    def test_attention_mode_widens_readout_and_requires_node_ids(self):
        predictor = self._predictor(readout="attention")

        self.assertEqual(predictor.readout, "attention")
        self.assertEqual(predictor.output[0].in_features, 16)  # 4 * hidden

        features = torch.randn(4, 2)
        edges = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
        with self.assertRaisesRegex(ValueError, "node_ids"):
            predictor(features, edges, torch.tensor([0, 1]))

    def test_attention_readout_runs_batched_and_trains(self):
        predictor = self._predictor(readout="attention")
        # Two graphs, each with a common neighbor of its targets:
        # 5-8-7 and 11-12-13 (nodes 8 / 12 touch both targets).
        features = torch.randn(6, 2)
        edges = torch.tensor([
            [0, 2, 1, 2, 3, 5, 4, 5],
            [2, 0, 2, 1, 5, 3, 5, 4],
        ])
        targets = torch.tensor([[0, 1], [3, 4]])  # second graph: rows 3..5
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        node_ids = torch.tensor([5, 7, 8, 11, 13, 12])

        logits = predictor(features, edges, targets, batch, node_ids=node_ids)
        self.assertEqual(logits.shape, (2, 7))

        logits.sum().backward()
        self.assertIsNotNone(predictor.link_attention.att.grad)
        self.assertTrue(
            torch.isfinite(predictor.link_attention.att.grad).all()
        )
        self.assertIsNotNone(predictor.ppr_encoder_cn[0].weight.grad)

    def test_attention_readout_requires_a_ppr_lookup(self):
        with self.assertRaisesRegex(ValueError, "PPRLookup"):
            PPIPredictor(
                esm_dim=2, hidden_dim=4, num_layers=1, heads=1,
                readout="attention",
            )
        with self.assertRaisesRegex(ValueError, "readout"):
            PPIPredictor(esm_dim=2, hidden_dim=4, num_layers=1, heads=1,
                         readout="bogus")


if __name__ == "__main__":
    unittest.main()
