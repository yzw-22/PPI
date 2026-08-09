"""Alternating optimization for the sampler and PPI predictor."""

import torch
from torch.nn import functional as F


class AlternatingTrainer:
    """Train a :class:`SubgraphSampler` and predictor with batched updates.

    The graph arguments must describe one dataset split.  The target edge is
    removed by ``SubgraphSampler`` before the trajectory is generated.
    """

    def __init__(self, sampler, predictor, sampler_optimizer, predictor_optimizer,
                 reinforce_baseline_coef=0.1, reinforce_gamma=1.0):
        if not 0.0 <= reinforce_gamma <= 1.0:
            raise ValueError("reinforce_gamma must be in [0, 1]")
        self.sampler = sampler
        self.predictor = predictor
        self.sampler_optimizer = sampler_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.reinforce_baseline_coef = reinforce_baseline_coef
        self.reinforce_gamma = reinforce_gamma

    def sampler_batch_step(self, node_features, edge_index, target_nodes, labels,
                           node_index=None):
        """Batch the predictor work while sampling each trajectory separately."""
        if target_nodes.shape[0] == 0:
            raise ValueError("target_nodes must not be empty")
        sampler_requires_grad = self._set_requires_grad(self.sampler, True)
        predictor_requires_grad = self._set_requires_grad(self.predictor, False)
        self.sampler.train()
        self.predictor.eval()
        adjacency = self.sampler._build_adjacency(
            edge_index, node_features.shape[0]
        )
        trajectories = [
            self.sampler.sample(
                node_features, edge_index, target, node_index,
                training=True, adjacency=adjacency
            )
            for target in target_nodes
        ]
        labels = labels.to(device=node_features.device, dtype=torch.float32)
        with torch.no_grad():
            baseline_logits = self._predict_graphs(
                node_features, [trajectory.baseline_graph for trajectory in trajectories]
            )
            baseline_losses = F.binary_cross_entropy_with_logits(
                baseline_logits, labels, reduction="none"
            ).mean(dim=1)

        step_records = [
            (sample_index, step)
            for sample_index, trajectory in enumerate(trajectories)
            for step in trajectory.steps
        ]
        if not step_records:
            self._restore_requires_grad(self.sampler, sampler_requires_grad)
            self._restore_requires_grad(self.predictor, predictor_requires_grad)
            zero = baseline_losses.new_zeros(())
            return {
                "sampler_loss": zero,
                "policy_loss": zero,
                "value_loss": zero,
                "baseline_loss": baseline_losses.mean().detach(),
                "mean_reward": zero,
            }

        with torch.no_grad():
            step_logits = self._predict_graphs(
                node_features, [step.graph for _, step in step_records]
            )
            step_labels = torch.stack([
                labels[sample_index] for sample_index, _ in step_records
            ])
            step_losses = F.binary_cross_entropy_with_logits(
                step_logits, step_labels, reduction="none"
            ).mean(dim=1)

        rewards_by_trajectory = [[] for _ in trajectories]
        record_index = 0
        for sample_index, trajectory in enumerate(trajectories):
            for _ in trajectory.steps:
                rewards_by_trajectory[sample_index].append(
                    (baseline_losses[sample_index] - step_losses[record_index]).detach()
                )
                record_index += 1

        returns_by_trajectory = [
            self._return_to_go(rewards, self.reinforce_gamma)
            for rewards in rewards_by_trajectory
        ]
        policy_losses = []
        value_losses = []
        rewards = []
        for sample_index, trajectory in enumerate(trajectories):
            for step_index, step in enumerate(trajectory.steps):
                reward = rewards_by_trajectory[sample_index][step_index]
                return_to_go = returns_by_trajectory[sample_index][step_index]
                advantage = (return_to_go - step.value).detach()
                rewards.append(reward)
                policy_losses.append(-step.log_prob * advantage)
                value_losses.append(F.mse_loss(step.value, return_to_go))

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        loss = policy_loss + self.reinforce_baseline_coef * value_loss
        self.sampler_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.sampler_optimizer.step()
        self._restore_requires_grad(self.sampler, sampler_requires_grad)
        self._restore_requires_grad(self.predictor, predictor_requires_grad)
        return {
            "sampler_loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "baseline_loss": baseline_losses.mean().detach(),
            "mean_reward": torch.stack(rewards).mean(),
        }

    def predictor_batch_step(self, node_features, edge_index, target_nodes, labels,
                             node_index=None):
        """Update the predictor on one final graph per greedy trajectory."""
        if target_nodes.shape[0] == 0:
            raise ValueError("target_nodes must not be empty")
        sampler_requires_grad = self._set_requires_grad(self.sampler, False)
        predictor_requires_grad = self._set_requires_grad(self.predictor, True)
        self.sampler.eval()
        self.predictor.train()
        adjacency = self.sampler._build_adjacency(
            edge_index, node_features.shape[0]
        )
        with torch.no_grad():
            trajectories = [
                self.sampler.sample(
                    node_features, edge_index, target, node_index,
                    training=False, adjacency=adjacency
                )
                for target in target_nodes
            ]
        labels = labels.to(device=node_features.device, dtype=torch.float32)
        # Match evaluation: train on the final graph only.  For a trajectory
        # with no actions, final_graph is initial_graph (including any virtual
        # proxy), rather than baseline_graph.
        graphs = [trajectory.final_graph for trajectory in trajectories]

        logits = self._predict_graphs(node_features, graphs)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="mean"
        )
        self.predictor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.predictor_optimizer.step()
        self._restore_requires_grad(self.sampler, sampler_requires_grad)
        self._restore_requires_grad(self.predictor, predictor_requires_grad)
        return {"predictor_loss": loss.detach()}

    def alternating_batch_step(self, node_features, edge_index, target_nodes,
                               labels, node_index=None):
        """Run one batched sampler update and one batched predictor update."""
        sampler_metrics = self.sampler_batch_step(
            node_features, edge_index, target_nodes, labels, node_index
        )
        predictor_metrics = self.predictor_batch_step(
            node_features, edge_index, target_nodes, labels, node_index
        )
        return {**sampler_metrics, **predictor_metrics}

    def _predict_graphs(self, node_features, graphs):
        features = []
        edges = []
        targets = []
        batches = []
        offset = 0
        for graph_index, graph in enumerate(graphs):
            graph_features = node_features[graph.feature_index]
            features.append(graph_features)
            edges.append(graph.edge_index + offset)
            targets.append(graph.target_nodes + offset)
            batches.append(torch.full(
                (graph_features.shape[0],), graph_index,
                device=graph_features.device, dtype=torch.long
            ))
            offset += graph_features.shape[0]
        return self.predictor(
            torch.cat(features),
            torch.cat(edges, dim=1),
            torch.stack(targets),
            torch.cat(batches),
        )

    @staticmethod
    def _return_to_go(rewards, gamma=1.0):
        """Compute discounted return-to-go for one trajectory."""
        if not rewards:
            return []
        returns = [None] * len(rewards)
        running = rewards[0].new_zeros(())
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            returns[index] = running
        return returns

    @staticmethod
    def _set_requires_grad(module, value):
        previous = [parameter.requires_grad for parameter in module.parameters()]
        for parameter in module.parameters():
            parameter.requires_grad_(value)
        return previous

    @staticmethod
    def _restore_requires_grad(module, previous):
        for parameter, requires_grad in zip(module.parameters(), previous):
            parameter.requires_grad_(requires_grad)
