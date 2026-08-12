import unittest

import torch

from src.predictor import PPIPredictor


class PPIPredictorTest(unittest.TestCase):
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
