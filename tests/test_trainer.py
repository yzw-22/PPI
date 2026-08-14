import unittest

import torch
from torch import nn

from src.predictor import PPIPredictor
from src.sampler import (
    RandomIterativeSubgraphSampler,
    RandomSubgraphSampler,
    SampledGraph,
    SamplingStep,
    SamplingTrajectory,
)
from src.train_shs27k import _aggregate_train_metrics
from src.trainer import AlternatingTrainer


class ReturnToGoTest(unittest.TestCase):
    def test_discounted_returns_are_computed_backward_per_trajectory(self):
        rewards = [
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
        ]

        returns = AlternatingTrainer._return_to_go(rewards, gamma=0.5)

        self.assertEqual(
            [float(value) for value in returns],
            [2.75, 3.5, 3.0],
        )

    def test_empty_trajectory_has_no_returns(self):
        self.assertEqual(AlternatingTrainer._return_to_go([], gamma=1.0), [])


class _RecordingSampler(nn.Module):
    def __init__(self, trajectory):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(0.0))
        self.trajectory = trajectory

    def _build_adjacency(self, edge_index, num_nodes):
        return [[] for _ in range(num_nodes)]

    def sample(self, *args, **kwargs):
        return self.trajectory


class _RecordingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))


class _RecordingTrainer(AlternatingTrainer):
    def _predict_graphs(self, node_features, graphs):
        self.seen_graphs = graphs
        return self.predictor.logit.expand(len(graphs), 7)


def _graph(feature_index):
    feature_index = torch.tensor(feature_index, dtype=torch.long)
    return SampledGraph(
        node_index=feature_index,
        feature_index=feature_index,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        target_nodes=torch.tensor([0, 1]),
    )


def _trainer_for(trajectory):
    sampler = _RecordingSampler(trajectory)
    predictor = _RecordingPredictor()
    return _RecordingTrainer(
        sampler,
        predictor,
        torch.optim.SGD(sampler.parameters(), lr=0.1),
        torch.optim.SGD(predictor.parameters(), lr=0.1),
    )


class PredictorGraphSelectionTest(unittest.TestCase):
    def test_predictor_update_uses_only_final_graph(self):
        trajectory = SamplingTrajectory(
            baseline_graph=_graph([0, 1]),
            steps=[
                SamplingStep(_graph([0, 1]), torch.tensor(0.0), torch.tensor(0.0)),
                SamplingStep(_graph([0, 1, 2]), torch.tensor(0.0), torch.tensor(0.0)),
            ],
        )
        trainer = _trainer_for(trajectory)

        trainer.predictor_batch_step(
            torch.zeros((3, 1)),
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([[0, 1]]),
            torch.zeros((1, 7)),
        )

        self.assertEqual(len(trainer.seen_graphs), 1)
        self.assertEqual(trainer.seen_graphs[0].feature_index.tolist(), [0, 1, 2])

    def test_predictor_update_uses_baseline_graph_when_no_step_exists(self):
        trajectory = SamplingTrajectory(
            baseline_graph=_graph([0, 1, 2]),
            steps=[],
        )
        trainer = _trainer_for(trajectory)

        trainer.predictor_batch_step(
            torch.zeros((3, 1)),
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([[0, 1]]),
            torch.zeros((1, 7)),
        )

        self.assertEqual(len(trainer.seen_graphs), 1)
        self.assertEqual(trainer.seen_graphs[0].feature_index.tolist(), [0, 1, 2])


class SamplerMetricAggregationTest(unittest.TestCase):
    def test_sampler_batch_reports_number_of_trajectory_steps(self):
        trajectory = SamplingTrajectory(
            baseline_graph=_graph([0, 1]),
            steps=[
                SamplingStep(_graph([0, 1]), torch.tensor(0.0), torch.tensor(0.0)),
                SamplingStep(_graph([0, 1, 2]), torch.tensor(0.0), torch.tensor(0.0)),
            ],
        )
        sampler = _RecordingSampler(trajectory)
        # Give the policy/value tensors a differentiable path so the sampler
        # update can run while this test inspects only the auxiliary count.
        for step in sampler.trajectory.steps:
            step.log_prob = sampler.parameter * 0
            step.value = sampler.parameter * 0
        predictor = _RecordingPredictor()
        trainer = _RecordingTrainer(
            sampler,
            predictor,
            torch.optim.SGD(sampler.parameters(), lr=0.1),
            torch.optim.SGD(predictor.parameters(), lr=0.1),
        )

        metrics = trainer.sampler_batch_step(
            torch.zeros((3, 1)),
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([[0, 1]]),
            torch.zeros((1, 7)),
        )

        self.assertEqual(metrics["sampler_step_count"], 2)

    def test_epoch_sampler_metrics_are_action_weighted(self):
        aggregate = _aggregate_train_metrics([
            (2, {
                "sampler_loss": 2.0,
                "mean_reward": 4.0,
                "predictor_loss": 10.0,
                "sampler_step_count": 1,
            }),
            (1, {
                "sampler_loss": 4.0,
                "mean_reward": 1.0,
                "predictor_loss": 2.0,
                "sampler_step_count": 3,
            }),
        ])

        self.assertAlmostEqual(aggregate["sampler_loss"], 3.5)
        self.assertAlmostEqual(aggregate["mean_reward"], 1.75)
        self.assertAlmostEqual(aggregate["predictor_loss"], 22 / 3)

    def test_no_action_batch_does_not_enter_sampler_denominator(self):
        aggregate = _aggregate_train_metrics([
            (2, {
                "sampler_loss": 99.0,
                "mean_reward": 99.0,
                "predictor_loss": 3.0,
                "sampler_step_count": 0,
            }),
            (1, {
                "sampler_loss": 5.0,
                "mean_reward": 7.0,
                "predictor_loss": 6.0,
                "sampler_step_count": 2,
            }),
        ])

        self.assertEqual(aggregate["sampler_loss"], 5.0)
        self.assertEqual(aggregate["mean_reward"], 7.0)
        self.assertEqual(aggregate["predictor_loss"], 4.0)

    def test_random_sampler_can_train_predictor_without_sampler_optimizer(self):
        sampler = RandomSubgraphSampler(esm_dim=2, max_context_nodes=10)
        predictor = PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, dropout=0.0
        )
        trainer = AlternatingTrainer(
            sampler,
            predictor,
            None,
            torch.optim.SGD(predictor.parameters(), lr=0.1),
        )

        metrics = trainer.predictor_batch_step(
            torch.randn((2, 2)),
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([[0, 1]]),
            torch.zeros((1, 7)),
        )

        self.assertIn("predictor_loss", metrics)

    def test_random_iterative_sampler_can_train_predictor_without_sampler_optimizer(self):
        sampler = RandomIterativeSubgraphSampler(
            esm_dim=2, fixed_num=0, max_steps=2, random_seed=42
        )
        predictor = PPIPredictor(
            esm_dim=2, hidden_dim=4, num_layers=1, heads=1, dropout=0.0
        )
        trainer = AlternatingTrainer(
            sampler,
            predictor,
            None,
            torch.optim.SGD(predictor.parameters(), lr=0.1),
        )

        metrics = trainer.predictor_batch_step(
            torch.randn((4, 2)),
            torch.tensor([[0, 1, 1], [1, 0, 2]]),
            torch.tensor([[0, 1]]),
            torch.zeros((1, 7)),
        )

        self.assertIn("predictor_loss", metrics)

if __name__ == "__main__":
    unittest.main()
