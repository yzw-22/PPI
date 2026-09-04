import unittest

import torch

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

    def test_predict_proba_public_compatibility_wrapper(self):
        predictor = PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, dropout=0.0
        )
        features = torch.randn(2, 2)
        edges = torch.empty((2, 0), dtype=torch.long)

        predictor.eval()
        with torch.no_grad():
            probabilities = predictor.predict_proba(
                features, edges, torch.tensor([0, 1])
            )

        self.assertEqual(probabilities.shape, (7,))
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))


if __name__ == "__main__":
    unittest.main()
