"""Alternating optimization for the sampler and PPI predictor."""

import torch
from torch.nn import functional as F


class AlternatingTrainer:
    """Train a :class:`SubgraphSampler` and predictor with batched updates.

    The graph arguments must describe one dataset split.  The target edge is
    removed by ``SubgraphSampler`` before the trajectory is generated.
    """

    def __init__(self, sampler, predictor, sampler_optimizer, predictor_optimizer,
                 reinforce_baseline_coef=0.1, reinforce_gamma=1.0,
                 complexity_penalty=0.0):
        if not 0.0 <= reinforce_gamma <= 1.0:
            raise ValueError("reinforce_gamma must be in [0, 1]")
        self.sampler = sampler
        self.predictor = predictor
        self.sampler_optimizer = sampler_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.reinforce_baseline_coef = reinforce_baseline_coef
        self.reinforce_gamma = reinforce_gamma
        self.complexity_penalty = float(complexity_penalty)

    def sampler_batch_step(self, node_features, edge_index, target_nodes, labels,
                           node_index=None, adjacency=None):
        """Batch the predictor work while sampling each trajectory separately.

        ``adjacency`` is a shared immutable split-level adjacency; when omitted
        it is built from ``edge_index`` each call.
        """
        if target_nodes.shape[0] == 0:
            raise ValueError("target_nodes must not be empty")
        sampler_requires_grad = self._set_requires_grad(self.sampler, True)
        predictor_requires_grad = self._set_requires_grad(self.predictor, False)
        self.sampler.train()
        self.predictor.eval()
        if adjacency is None:
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
                "sampler_step_count": 0,
                **self._trajectory_diagnostics(trajectories, baseline_losses.mean(),
                                               baseline_losses.mean(), zero),
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
            step_count = len(trajectory.steps)
            trajectory_step_losses = step_losses[
                record_index:record_index + step_count
            ]
            rewards_by_trajectory[sample_index] = self._trajectory_rewards(
                trajectory,
                baseline_losses[sample_index],
                trajectory_step_losses,
                self.complexity_penalty,
            )
            record_index += step_count

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
        final_losses = []
        record_index = 0
        for sample_index, trajectory in enumerate(trajectories):
            if trajectory.steps:
                final_losses.append(step_losses[record_index + len(trajectory.steps) - 1])
                record_index += len(trajectory.steps)
            else:
                final_losses.append(baseline_losses[sample_index])
        final_loss = torch.stack(final_losses).mean()
        return {
            "sampler_loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "baseline_loss": baseline_losses.mean().detach(),
            "mean_reward": torch.stack(rewards).mean(),
            "sampler_step_count": len(step_records),
            **self._trajectory_diagnostics(
                trajectories, baseline_losses.mean(), final_loss,
                torch.stack(rewards).gt(0).float().mean(),
            ),
        }

    def predictor_batch_step(self, node_features, edge_index, target_nodes, labels,
                             node_index=None, adjacency=None):
        """Update the predictor on one final graph per greedy trajectory."""
        if target_nodes.shape[0] == 0:
            raise ValueError("target_nodes must not be empty")
        sampler_requires_grad = self._set_requires_grad(self.sampler, False)
        predictor_requires_grad = self._set_requires_grad(self.predictor, True)
        self.sampler.eval()
        self.predictor.train()
        if adjacency is None:
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
        # Match evaluation: train on the final graph only. For a trajectory
        # with no actions, final_graph is the G0 baseline graph.
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
        return {
            "predictor_loss": loss.detach(),
            **self._trajectory_diagnostics(
                trajectories, loss.detach(), loss.detach(),
                torch.tensor(0.0, device=loss.device),
            ),
        }

    @staticmethod
    def _trajectory_diagnostics(trajectories, baseline_loss, final_loss,
                                reward_positive_rate):
        count = max(1, len(trajectories))
        device = baseline_loss.device
        return {
            "mean_baseline_loss": baseline_loss.detach(),
            "mean_final_loss": final_loss.detach(),
            "mean_steps": torch.tensor(
                sum(len(t.steps) for t in trajectories) / count, device=device
            ),
            "mean_final_nodes": torch.tensor(
                sum(t.final_graph.node_index.numel() for t in trajectories) / count,
                device=device,
            ),
            "mean_context_nodes": torch.tensor(
                sum(t.context_node_count for t in trajectories) / count,
                device=device,
            ),
            "mean_proxy_count": torch.tensor(
                sum(len(t.proxy_nodes) for t in trajectories) / count, device=device
            ),
            "mean_real_edges": torch.tensor(
                sum(t.final_graph.real_edge_count for t in trajectories) / count,
                device=device,
            ),
            "stop_rate": torch.tensor(
                sum(bool(t.stopped) for t in trajectories) / count, device=device
            ),
            "reward_positive_rate": reward_positive_rate.detach(),
        }

    def alternating_batch_step(self, node_features, edge_index, target_nodes,
                               labels, node_index=None, adjacency=None):
        """Run one batched sampler update and one batched predictor update."""
        sampler_metrics = self.sampler_batch_step(
            node_features, edge_index, target_nodes, labels, node_index,
            adjacency=adjacency,
        )
        predictor_metrics = self.predictor_batch_step(
            node_features, edge_index, target_nodes, labels, node_index,
            adjacency=adjacency,
        )
        # Sampler pass computes the trajectory-level diagnostics (including
        # baseline versus final loss).  The predictor pass only contributes
        # its sample-weighted optimization loss; replacing the diagnostics
        # here would silently report the predictor-pass values instead.
        return {**sampler_metrics, "predictor_loss": predictor_metrics["predictor_loss"]}

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
    def _trajectory_rewards(trajectory, baseline_loss, step_losses,
                            complexity_penalty=0.0):
        """Compute incremental rewards for one sampled trajectory.

        A regular expansion is rewarded by the predictor-loss improvement over
        the immediately preceding graph.  STOP is a terminal decision that
        does not change the graph, so it receives no duplicate graph reward.
        Complexity is charged only for nodes newly added by the current
        expansion, rather than repeatedly charging all nodes added so far.
        """
        if len(trajectory.steps) != len(step_losses):
            raise ValueError("step_losses must match trajectory.steps")

        baseline_node_count = int(trajectory.baseline_graph.node_index.numel())
        previous_loss = baseline_loss
        previous_extra_nodes = 0
        rewards = []
        for step, current_loss in zip(trajectory.steps, step_losses):
            current_extra_nodes = max(
                0,
                int(step.graph.node_index.numel()) - baseline_node_count,
            )
            if step.is_stop:
                reward = current_loss.new_zeros(())
            else:
                new_extra_nodes = max(
                    0, current_extra_nodes - previous_extra_nodes
                )
                reward = (
                    previous_loss - current_loss
                    - complexity_penalty * new_extra_nodes
                )
            rewards.append(reward.detach())
            previous_loss = current_loss
            previous_extra_nodes = current_extra_nodes
        return rewards

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
