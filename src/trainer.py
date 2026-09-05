"""Alternating optimization for the sampler and PPI predictor."""

import torch
from torch.nn import functional as F


class AlternatingTrainer:
    """Train a :class:`SubgraphSampler` and predictor with batched updates.

    The graph arguments must describe one shared knowledge graph (the full
    dataset graph in the training entry).  The target edge is removed by
    ``SubgraphSampler`` before the trajectory is generated.
    """

    def __init__(self, sampler, predictor, sampler_optimizer, predictor_optimizer,
                 reinforce_gamma=1.0, edge_relations=None, reward="bce_diff",
                 reward_pos=2.0, reward_neg=1.0, reward_ref="initial"):
        if not 0.0 <= reinforce_gamma <= 1.0:
            raise ValueError("reinforce_gamma must be in [0, 1]")
        if reward not in ("bce_diff", "margin"):
            raise ValueError("reward must be 'bce_diff' or 'margin'")
        if reward_pos <= 0.0 or reward_neg <= 0.0:
            raise ValueError("reward_pos and reward_neg must be positive")
        if reward_ref not in ("initial", "base"):
            raise ValueError("reward_ref must be 'initial' or 'base'")
        # ``sampler_optimizer`` may be None when the sampler has no parameters
        # (e.g. the non-learnable ``StaticNeighborhoodSampler``) or follows a
        # non-learnable uniform policy: its update phase is a no-op either
        # because such trajectories carry no steps or because the policy is
        # not scored by a network.
        self.sampler = sampler
        self.predictor = predictor
        self.sampler_optimizer = sampler_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.reinforce_gamma = reinforce_gamma
        self.edge_relations = edge_relations
        self.reward = reward
        self.reward_pos = reward_pos
        self.reward_neg = reward_neg
        self.reward_ref = reward_ref

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
                training=True, adjacency=adjacency,
                edge_relations=self.edge_relations,
            )
            for target in target_nodes
        ]
        labels = labels.to(device=node_features.device, dtype=torch.float32)
        # The reward reference forward. With ``reward_ref="base"`` a
        # trajectory carrying an augmented static base is scored against its
        # base-only prediction — the marginal value of the RL additions —
        # while ``reward_ref="initial"`` keeps the historical G0 reference
        # bit-for-bit (the metric keys keep the historical ``baseline_`` name).
        reference_graphs = [
            trajectory.reference_graph
            if self.reward_ref == "base"
            and trajectory.reference_graph is not None
            else trajectory.baseline_graph
            for trajectory in trajectories
        ]
        with torch.no_grad():
            baseline_logits = self._predict_graphs(
                node_features, reference_graphs
            )
            baseline_losses = F.binary_cross_entropy_with_logits(
                baseline_logits, labels, reduction="none"
            ).mean(dim=1)
            baseline_margins = self._label_margins(baseline_logits, labels)

        step_records = [
            (sample_index, step)
            for sample_index, trajectory in enumerate(trajectories)
            for step in trajectory.steps
        ]
        if not step_records or self.sampler_optimizer is None:
            # Non-learnable policies (step-free static-style trajectories, or
            # a uniform control policy with steps) contribute no sampler
            # update: report the reference metrics with zero losses. The
            # zero-step final-margin fallback reads the reference margins,
            # i.e. the base margin for an augmented trajectory.
            self._restore_requires_grad(self.sampler, sampler_requires_grad)
            self._restore_requires_grad(self.predictor, predictor_requires_grad)
            zero = baseline_losses.new_zeros(())
            return {
                "sampler_loss": zero,
                "policy_loss": zero,
                "baseline_loss": baseline_losses.mean().detach(),
                "mean_reward": zero,
                "mean_final_margin": baseline_margins.mean().detach(),
                "sampler_step_count": len(step_records),
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
            step_margins = self._label_margins(step_logits, step_labels)

        rewards_by_trajectory = []
        final_margins = []
        record_index = 0
        for sample_index, trajectory in enumerate(trajectories):
            step_count = len(trajectory.steps)
            margins = step_margins[record_index:record_index + step_count]
            if self.reward == "margin":
                rewards_by_trajectory.append(self._trajectory_rewards_margin(
                    baseline_margins[sample_index], margins,
                    self.reward_pos, self.reward_neg,
                ))
            else:
                rewards_by_trajectory.append(self._trajectory_rewards(
                    baseline_losses[sample_index],
                    step_losses[record_index:record_index + step_count],
                ))
            if step_count:
                final_margins.append(margins[-1])
            else:
                final_margins.append(baseline_margins[sample_index])
            record_index += step_count

        returns_by_trajectory = [
            self._return_to_go(rewards, self.reinforce_gamma)
            for rewards in rewards_by_trajectory
        ]
        raw_advantages = []
        rewards = []
        for sample_index, trajectory in enumerate(trajectories):
            for step_index, step in enumerate(trajectory.steps):
                reward = rewards_by_trajectory[sample_index][step_index]
                return_to_go = returns_by_trajectory[sample_index][step_index]
                rewards.append(reward)
                # No learned baseline: the discounted return-to-go is the
                # advantage; batch standardization below recenters it.
                raw_advantages.append(return_to_go.detach())

        advantages = self._normalize_advantages(torch.stack(raw_advantages))
        policy_losses = [
            -step.log_prob * advantage
            for (_, step), advantage in zip(step_records, advantages)
        ]
        policy_loss = torch.stack(policy_losses).mean()
        loss = policy_loss
        self.sampler_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.sampler_optimizer.step()
        self._restore_requires_grad(self.sampler, sampler_requires_grad)
        self._restore_requires_grad(self.predictor, predictor_requires_grad)
        return {
            "sampler_loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "baseline_loss": baseline_losses.mean().detach(),
            "mean_reward": torch.stack(rewards).mean(),
            "mean_final_margin": torch.stack(final_margins).mean().detach(),
            "sampler_step_count": len(step_records),
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
                    training=False, adjacency=adjacency,
                    edge_relations=self.edge_relations,
                )
                for target in target_nodes
            ]
        labels = labels.to(device=node_features.device, dtype=torch.float32)
        # Match evaluation: train on the prediction graph only. A trajectory
        # with actions ends at its last action graph; a step-free augmented
        # trajectory predicts on its static base (``reference_graph``), and a
        # plain step-free one on the G0 baseline graph.
        graphs = [trajectory.prediction_graph for trajectory in trajectories]

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
        return {**sampler_metrics, **predictor_metrics}

    def _predict_graphs(self, node_features, graphs):
        features = []
        edges = []
        targets = []
        batches = []
        edge_attrs = []
        node_ids = []
        edge_dim = getattr(self.predictor, "edge_dim", None)
        offset = 0
        for graph_index, graph in enumerate(graphs):
            graph_features = node_features[graph.feature_index]
            features.append(graph_features)
            edges.append(graph.edge_index + offset)
            node_ids.append(graph.node_index)
            if edge_dim is not None:
                if graph.edge_attr is None:
                    raise ValueError(
                        "sampled graphs must provide edge_attr for a relation-aware predictor"
                    )
                edge_attrs.append(graph.edge_attr)
            targets.append(graph.target_nodes + offset)
            batches.append(torch.full(
                (graph_features.shape[0],), graph_index,
                device=graph_features.device, dtype=torch.long
            ))
            offset += graph_features.shape[0]
        predictor_args = (
            torch.cat(features),
            torch.cat(edges, dim=1),
            torch.stack(targets),
            torch.cat(batches),
        )
        if getattr(self.predictor, "readout", "mean") == "attention":
            node_ids_arg = torch.cat(node_ids)
            if edge_dim is None:
                return self.predictor(*predictor_args, node_ids=node_ids_arg)
            return self.predictor(
                *predictor_args, edge_attr=torch.cat(edge_attrs, dim=0),
                node_ids=node_ids_arg,
            )
        if edge_dim is None:
            return self.predictor(*predictor_args)
        return self.predictor(
            *predictor_args, edge_attr=torch.cat(edge_attrs, dim=0)
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
    def _label_margins(logits, labels):
        """Label-aligned mean probability margin per sample.

        ``M(p) = mean_j((2*y_j - 1) * p_j) ∈ [-1, 1]``: positive exactly when
        the predicted probabilities lean toward the multi-hot label vector.
        """
        signs = labels * 2.0 - 1.0
        return (signs * torch.sigmoid(logits)).mean(dim=1)

    @staticmethod
    def _trajectory_rewards_margin(baseline_margin, step_margins,
                                   w_pos=2.0, w_neg=1.0):
        """Margin-improvement rewards against the fixed reference graph.

        Every step is scored by ``M(step) - M(ref)`` — the same reference for
        all steps (G0, or the augmented static base under
        ``reward_ref="base"``), unlike the sequential BCE difference — and
        scaled by ``w_pos`` for improvements and ``w_neg`` for regressions
        (RISE-DDI-style asymmetric weighting).  The margin is linear and
        bounded in [−1, 1], avoiding the heavy tails of BCE differences.
        """
        delta = step_margins - baseline_margin
        scaled = torch.where(delta > 0, delta * w_pos, delta * w_neg)
        return [value.detach() for value in scaled]

    @staticmethod
    def _trajectory_rewards(baseline_loss, step_losses):
        """Return detached loss-improvement rewards for one trajectory.

        The first action is compared with ``G0``; every later action is
        compared only with the immediately preceding sampled graph.
        """
        previous_loss = baseline_loss
        rewards = []
        for current_loss in step_losses:
            rewards.append((previous_loss - current_loss).detach())
            previous_loss = current_loss
        return rewards

    @staticmethod
    def _normalize_advantages(advantages, eps=1e-8):
        """Standardize detached advantages across all action steps in a batch."""
        if advantages.numel() == 0:
            return advantages
        return (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(eps)

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
