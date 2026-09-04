import unittest

import torch
from torch import nn

from src.sampler import (
    RandomSubsetSampler,
    SampledGraph,
    SamplingStep,
    SamplingTrajectory,
    StaticNeighborhoodSampler,
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


class RewardAndAdvantageTest(unittest.TestCase):
    def test_rewards_compare_each_step_with_the_previous_graph(self):
        rewards = AlternatingTrainer._trajectory_rewards(
            torch.tensor(1.0),
            torch.tensor([0.8, 0.9, 0.6]),
        )

        for reward, expected in zip(rewards, [0.2, -0.1, 0.3]):
            self.assertAlmostEqual(float(reward), expected, places=6)

    def test_batch_advantages_are_standardized_with_population_std(self):
        normalized = AlternatingTrainer._normalize_advantages(
            torch.tensor([1.0, 3.0, 5.0])
        )

        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std(unbiased=False)), 1.0, places=6)

    def test_constant_batch_advantages_remain_finite(self):
        normalized = AlternatingTrainer._normalize_advantages(
            torch.tensor([2.0, 2.0])
        )

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))


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
                SamplingStep(_graph([0, 1]), torch.tensor(0.0)),
                SamplingStep(_graph([0, 1, 2]), torch.tensor(0.0)),
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


class StaticSamplerTrainerIntegrationTest(unittest.TestCase):
    def test_sampler_update_is_a_no_op_and_predictor_trains_on_the_static_graph(self):
        # Edges: 0-1 (target, removed) and 0-2 (safe). The static 1-hop region
        # of {0, 1} is {0, 1, 2}.
        node_features = torch.zeros((3, 2))
        edge_index = torch.tensor([
            [0, 1, 0, 2],
            [1, 0, 2, 0],
        ])
        sampler = StaticNeighborhoodSampler(esm_dim=2, k_hops=1)
        predictor = _RecordingPredictor()
        trainer = _RecordingTrainer(
            sampler,
            predictor,
            None,  # the static sampler has no parameters
            torch.optim.SGD(predictor.parameters(), lr=0.1),
        )

        sampler_metrics = trainer.sampler_batch_step(
            node_features, edge_index, torch.tensor([[0, 1]]), torch.zeros((1, 7))
        )
        self.assertEqual(sampler_metrics["sampler_step_count"], 0)
        self.assertEqual(float(sampler_metrics["sampler_loss"]), 0.0)

        trainer.predictor_batch_step(
            node_features, edge_index, torch.tensor([[0, 1]]), torch.zeros((1, 7))
        )
        self.assertEqual(len(trainer.seen_graphs), 1)
        self.assertEqual(
            sorted(trainer.seen_graphs[0].feature_index.tolist()), [0, 1, 2]
        )


class RandomSubsetSamplerTrainerIntegrationTest(unittest.TestCase):
    def test_sampler_update_is_a_no_op_and_predictor_trains_on_the_random_graph(self):
        # Edges: 0-1 (target, removed) and 0-2 (safe). The region of {0, 1} is
        # {0, 1, 2}; with min_size 3..4 the random subset always takes all of it.
        node_features = torch.zeros((3, 2))
        edge_index = torch.tensor([
            [0, 1, 0, 2],
            [1, 0, 2, 0],
        ])
        torch.manual_seed(0)
        sampler = RandomSubsetSampler(esm_dim=2, k_hops=1, min_size=3, max_size=4)
        predictor = _RecordingPredictor()
        trainer = _RecordingTrainer(
            sampler,
            predictor,
            None,  # the random-subset sampler has no parameters
            torch.optim.SGD(predictor.parameters(), lr=0.1),
        )

        sampler_metrics = trainer.sampler_batch_step(
            node_features, edge_index, torch.tensor([[0, 1]]), torch.zeros((1, 7))
        )
        self.assertEqual(sampler_metrics["sampler_step_count"], 0)
        self.assertEqual(float(sampler_metrics["sampler_loss"]), 0.0)

        trainer.predictor_batch_step(
            node_features, edge_index, torch.tensor([[0, 1]]), torch.zeros((1, 7))
        )
        self.assertEqual(len(trainer.seen_graphs), 1)
        self.assertEqual(
            sorted(trainer.seen_graphs[0].feature_index.tolist()), [0, 1, 2]
        )


class SamplerMetricAggregationTest(unittest.TestCase):
    def test_sampler_batch_reports_number_of_trajectory_steps(self):
        trajectory = SamplingTrajectory(
            baseline_graph=_graph([0, 1]),
            steps=[
                SamplingStep(_graph([0, 1]), torch.tensor(0.0)),
                SamplingStep(_graph([0, 1, 2]), torch.tensor(0.0)),
            ],
        )
        sampler = _RecordingSampler(trajectory)
        # Give the policy tensor a differentiable path so the sampler update
        # can run while this test inspects only the auxiliary count.
        for step in sampler.trajectory.steps:
            step.log_prob = sampler.parameter * 0
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
                "mean_final_margin": 0.2,
                "sampler_step_count": 1,
            }),
            (1, {
                "sampler_loss": 4.0,
                "mean_reward": 1.0,
                "predictor_loss": 2.0,
                "mean_final_margin": 0.5,
                "sampler_step_count": 3,
            }),
        ])

        self.assertAlmostEqual(aggregate["sampler_loss"], 3.5)
        self.assertAlmostEqual(aggregate["mean_reward"], 1.75)
        self.assertAlmostEqual(aggregate["predictor_loss"], 22 / 3)
        self.assertAlmostEqual(aggregate["mean_final_margin"], 0.3)

    def test_no_action_batch_does_not_enter_sampler_denominator(self):
        aggregate = _aggregate_train_metrics([
            (2, {
                "sampler_loss": 99.0,
                "mean_reward": 99.0,
                "predictor_loss": 3.0,
                "mean_final_margin": 0.1,
                "sampler_step_count": 0,
            }),
            (1, {
                "sampler_loss": 5.0,
                "mean_reward": 7.0,
                "predictor_loss": 6.0,
                "mean_final_margin": 0.4,
                "sampler_step_count": 2,
            }),
        ])

        self.assertEqual(aggregate["sampler_loss"], 5.0)
        self.assertEqual(aggregate["mean_reward"], 7.0)
        self.assertEqual(aggregate["predictor_loss"], 4.0)
        self.assertAlmostEqual(aggregate["mean_final_margin"], 0.2)


class MarginRewardTest(unittest.TestCase):
    def test_label_margins_align_with_the_label_vector(self):
        logits = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
        labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        margins = AlternatingTrainer._label_margins(logits, labels)

        expected_0 = (float(torch.sigmoid(torch.tensor(2.0)))
                      - float(torch.sigmoid(torch.tensor(-2.0)))) / 2.0
        self.assertAlmostEqual(float(margins[0]), expected_0, places=6)
        self.assertAlmostEqual(float(margins[1]), 0.0, places=6)

    def test_margin_rewards_scale_around_the_fixed_reference(self):
        rewards = AlternatingTrainer._trajectory_rewards_margin(
            torch.tensor(0.2),
            torch.tensor([0.3, 0.1, 0.2]),
            w_pos=2.0, w_neg=1.0,
        )

        self.assertAlmostEqual(float(rewards[0]), 0.2, places=6)
        self.assertAlmostEqual(float(rewards[1]), -0.1, places=6)
        self.assertAlmostEqual(float(rewards[2]), 0.0, places=6)
        for reward in rewards:
            self.assertFalse(reward.requires_grad)

    def test_trainer_validates_reward_configuration(self):
        sampler = _RecordingSampler(
            SamplingTrajectory(baseline_graph=_graph([0, 1]), steps=[])
        )
        predictor = _RecordingPredictor()
        for kwargs in (
            {"reward": "bogus"},
            {"reward_pos": 0.0},
            {"reward_neg": -1.0},
        ):
            with self.assertRaisesRegex(ValueError, "reward"):
                AlternatingTrainer(
                    sampler, predictor,
                    torch.optim.SGD(sampler.parameters(), lr=0.1),
                    torch.optim.SGD(predictor.parameters(), lr=0.1),
                    **kwargs,
                )

    def test_margin_mode_reports_final_margins_and_zero_policy_loss(self):
        # Static sampler: no steps, so the final margin is the baseline
        # margin. _RecordingPredictor emits logit 0 -> sigmoid 0.5; with an
        # all-zero label vector every sign is -1 -> M = -0.5.
        node_features = torch.zeros((3, 2))
        edge_index = torch.tensor([
            [0, 1, 0, 2],
            [1, 0, 2, 0],
        ])
        sampler = StaticNeighborhoodSampler(esm_dim=2, k_hops=1)
        predictor = _RecordingPredictor()
        trainer = _RecordingTrainer(
            sampler,
            predictor,
            None,
            torch.optim.SGD(predictor.parameters(), lr=0.1),
            reward="margin",
        )

        metrics = trainer.sampler_batch_step(
            node_features, edge_index, torch.tensor([[0, 1]]),
            torch.zeros((1, 7)),
        )

        self.assertEqual(metrics["sampler_step_count"], 0)
        self.assertAlmostEqual(float(metrics["mean_final_margin"]), -0.5,
                               places=6)
        self.assertAlmostEqual(float(metrics["sampler_loss"]), 0.0, places=6)

if __name__ == "__main__":
    unittest.main()
