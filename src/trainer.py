"""Alternating optimization for the sampler and PPI predictor."""

import torch
from torch.nn import functional as F


class AlternatingTrainer:
    """Train a :class:`SubgraphSampler` and predictor one target pair at a time.

    The graph arguments must describe one dataset split.  The target edge is
    removed by ``SubgraphSampler`` before the trajectory is generated.
    """

    def __init__(self, sampler, predictor, sampler_optimizer, predictor_optimizer,
                 reinforce_baseline_coef=0.1):
        self.sampler = sampler
        self.predictor = predictor
        self.sampler_optimizer = sampler_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.reinforce_baseline_coef = reinforce_baseline_coef

    def sampler_step(self, node_features, edge_index, target_nodes, label,
                     node_index=None):
        """Update the sampler while keeping the predictor fixed."""
        sampler_requires_grad = self._set_requires_grad(self.sampler, True)
        predictor_requires_grad = self._set_requires_grad(self.predictor, False)
        self.sampler.train()
        self.predictor.eval()

        trajectory = self.sampler.sample(
            node_features, edge_index, target_nodes, node_index, training=True
        )
        label = label.to(device=node_features.device, dtype=torch.float32)
        with torch.no_grad():
            baseline_logits = self._predict_graph(
                node_features, trajectory.baseline_graph
            )
            baseline_loss = F.binary_cross_entropy_with_logits(
                baseline_logits, label
            )

        if not trajectory.steps:
            self._restore_requires_grad(self.sampler, sampler_requires_grad)
            self._restore_requires_grad(self.predictor, predictor_requires_grad)
            return {
                "sampler_loss": baseline_loss.new_zeros(()),
                "policy_loss": baseline_loss.new_zeros(()),
                "value_loss": baseline_loss.new_zeros(()),
                "baseline_loss": baseline_loss.detach(),
                "mean_reward": baseline_loss.new_zeros(()),
            }

        rewards = []
        policy_losses = []
        value_losses = []
        for step in trajectory.steps:
            with torch.no_grad():
                logits = self._predict_graph(node_features, step.graph)
                step_loss = F.binary_cross_entropy_with_logits(logits, label)
            reward = baseline_loss - step_loss
            rewards.append(reward.detach())
            advantage = reward.detach() - step.value
            policy_losses.append(-step.log_prob * advantage.detach())
            value_losses.append(F.mse_loss(step.value, reward.detach()))

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
            "baseline_loss": baseline_loss.detach(),
            "mean_reward": torch.stack(rewards).mean(),
        }

    def predictor_step(self, node_features, edge_index, target_nodes, label,
                       node_index=None):
        """Update the predictor using the sampler's greedy trajectory."""
        sampler_requires_grad = self._set_requires_grad(self.sampler, False)
        predictor_requires_grad = self._set_requires_grad(self.predictor, True)
        self.sampler.eval()
        self.predictor.train()

        with torch.no_grad():
            trajectory = self.sampler.sample(
                node_features, edge_index, target_nodes, node_index, training=False
            )
        graphs = [step.graph for step in trajectory.steps]
        if not graphs:
            graphs = [trajectory.baseline_graph]

        label = label.to(device=node_features.device, dtype=torch.float32)
        losses = [
            F.binary_cross_entropy_with_logits(
                self._predict_graph(node_features, graph), label
            )
            for graph in graphs
        ]
        loss = torch.stack(losses).mean()
        self.predictor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.predictor_optimizer.step()

        self._restore_requires_grad(self.sampler, sampler_requires_grad)
        self._restore_requires_grad(self.predictor, predictor_requires_grad)
        return {"predictor_loss": loss.detach()}

    def sampler_batch_step(self, node_features, edge_index, target_nodes, labels,
                           node_index=None):
        """Batch the predictor work while sampling each trajectory separately."""
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

        policy_losses = []
        value_losses = []
        rewards = []
        for record_index, (sample_index, step) in enumerate(step_records):
            reward = baseline_losses[sample_index] - step_losses[record_index]
            rewards.append(reward.detach())
            policy_losses.append(-step.log_prob * (reward.detach() - step.value).detach())
            value_losses.append(F.mse_loss(step.value, reward.detach()))

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
        """Update the predictor on greedy trajectories from one batch."""
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
        graphs = []
        graph_labels = []
        labels = labels.to(device=node_features.device, dtype=torch.float32)
        for sample_index, trajectory in enumerate(trajectories):
            sampled_graphs = [step.graph for step in trajectory.steps]
            if not sampled_graphs:
                sampled_graphs = [trajectory.baseline_graph]
            graphs.extend(sampled_graphs)
            graph_labels.extend([labels[sample_index]] * len(sampled_graphs))

        logits = self._predict_graphs(node_features, graphs)
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.stack(graph_labels), reduction="mean"
        )
        self.predictor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.predictor_optimizer.step()
        self._restore_requires_grad(self.sampler, sampler_requires_grad)
        self._restore_requires_grad(self.predictor, predictor_requires_grad)
        return {"predictor_loss": loss.detach()}

    def alternating_step(self, node_features, edge_index, target_nodes, label,
                         node_index=None):
        """Run one sampler update followed by one predictor update."""
        sampler_metrics = self.sampler_step(
            node_features, edge_index, target_nodes, label, node_index
        )
        predictor_metrics = self.predictor_step(
            node_features, edge_index, target_nodes, label, node_index
        )
        return {**sampler_metrics, **predictor_metrics}

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

    def _predict_graph(self, node_features, graph):
        return self.predictor(
            node_features[graph.feature_index],
            graph.edge_index,
            graph.target_nodes,
        )

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
    def _set_requires_grad(module, value):
        previous = [parameter.requires_grad for parameter in module.parameters()]
        for parameter in module.parameters():
            parameter.requires_grad_(value)
        return previous

    @staticmethod
    def _restore_requires_grad(module, previous):
        for parameter, requires_grad in zip(module.parameters(), previous):
            parameter.requires_grad_(requires_grad)
