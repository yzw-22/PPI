"""Run an alternating-training PPI experiment.

Supports the SHS27k / SHS148k / STRING datasets via ``--dataset`` and their
available bfs / dfs / random splits via ``--split`` (the valid split choices
depend on the dataset; see ``PPIGraph.AVAILABLE_SPLITS``). The split file
``{root}/{dataset}_{split}.json`` is checked at argument-parse time; the
remaining dataset files are assumed to exist under ``--root`` and are not
pre-checked. Note that bfs is only shipped for SHS27k under the default
``dataset`` root (SHS148k bfs lives under ``dataset_ppisplit``).
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .ppi_graph import PPIGraph
from .ppr import PPRLookup
from .predictor import PPIPredictor
from .sampler import (
    EdgeRelationLookup,
    HeuristicSampler,
    RandomSubsetSampler,
    StaticNeighborhoodSampler,
    SubgraphSampler,
)
from .trainer import AlternatingTrainer


def _metrics(labels, probabilities):
    """Return multi-label metrics for one evaluation subset.

    A subset whose ROC-AUC is undefined reports ``None`` (the same convention
    as an empty group) instead of ``NaN``: Macro-AUC is undefined when any
    class has a single label value, and Micro-AUC is undefined when no
    positive/negative example exists at all.
    """
    if len(labels) == 0:
        return {
            "count": 0,
            "roc_auc_macro": None,
            "roc_auc_micro": None,
            "f1_macro": None,
            "f1_micro": None,
        }
    predictions = probabilities >= 0.5
    constant_class = [
        np.unique(labels[:, cls_idx]).size < 2
        for cls_idx in range(labels.shape[1])
    ]
    return {
        "count": int(len(labels)),
        "roc_auc_macro": (
            None
            if any(constant_class)
            else float(roc_auc_score(labels, probabilities, average="macro"))
        ),
        "roc_auc_micro": (
            None
            if np.unique(labels).size < 2
            else float(roc_auc_score(labels, probabilities, average="micro"))
        ),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
    }


def evaluate(trainer, node_features, graph, targets, labels, batch_size,
             training_node_index, adjacency=None):
    """Evaluate final graphs and group pairs by training-node visibility.

    ``graph`` is the shared knowledge graph dict (the full dataset graph in
    the training entry).  ``adjacency`` is its shared immutable graph-level
    adjacency; when omitted it is rebuilt from ``graph`` each call so the
    function stays self-contained.
    """
    trainer.sampler.eval()
    trainer.predictor.eval()
    if adjacency is None:
        adjacency = trainer.sampler._build_adjacency(
            graph["edge_index"], node_features.shape[0]
        )
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            target_batch = targets[start:start + batch_size]
            trajectories = [
                trainer.sampler.sample(
                    node_features,
                    graph["edge_index"],
                    target,
                    graph["node_index"],
                    training=False,
                    adjacency=adjacency,
                    edge_relations=trainer.edge_relations,
                )
                for target in target_batch
            ]
            graphs = [trajectory.prediction_graph for trajectory in trajectories]
            logits = trainer._predict_graphs(node_features, graphs)
            probabilities.append(torch.sigmoid(logits).cpu())

    probabilities = torch.cat(probabilities).numpy()
    labels = labels.cpu().numpy()
    metrics = _metrics(labels, probabilities)

    # Both tensors are integer indices (never require grad), so ``.detach()``
    # would be a no-op here.
    training_nodes = set(training_node_index.cpu().tolist())
    groups = {"BS": [], "ES": [], "NS": []}
    for sample_index, (source, target) in enumerate(targets.cpu().tolist()):
        source_seen = source in training_nodes
        target_seen = target in training_nodes
        group = "BS" if source_seen and target_seen else (
            "ES" if source_seen or target_seen else "NS"
        )
        groups[group].append(sample_index)
    metrics["visibility"] = {
        group: _metrics(labels[indices], probabilities[indices])
        for group, indices in groups.items()
    }
    return metrics


def _aggregate_train_metrics(batch_records):
    """Aggregate one epoch using each trainer metric's native denominator.

    Sampler metrics returned by ``AlternatingTrainer`` are means over trajectory
    steps, so every action receives equal weight here.  Predictor loss is a
    mean over PPI samples and remains sample-weighted.
    """
    sampler_loss_total = 0.0
    reward_total = 0.0
    sampler_step_count = 0
    predictor_loss_total = 0.0
    margin_total = 0.0
    sample_count = 0
    for batch_size, metrics in batch_records:
        steps = int(metrics["sampler_step_count"])
        sampler_loss_total += float(metrics["sampler_loss"]) * steps
        reward_total += float(metrics["mean_reward"]) * steps
        sampler_step_count += steps

        sampler_size = int(batch_size)
        predictor_loss_total += float(metrics["predictor_loss"]) * sampler_size
        margin_total += float(metrics["mean_final_margin"]) * sampler_size
        sample_count += sampler_size

    return {
        "sampler_loss": (
            sampler_loss_total / sampler_step_count
            if sampler_step_count else 0.0
        ),
        "predictor_loss": (
            predictor_loss_total / sample_count
            if sample_count else 0.0
        ),
        "mean_reward": (
            reward_total / sampler_step_count
            if sampler_step_count else 0.0
        ),
        "mean_final_margin": (
            margin_total / sample_count
            if sample_count else 0.0
        ),
        "sampler_step_count": sampler_step_count,
    }


def _build_training_edge_relations(graph, full_node_index):
    """Return local-id relation features sourced exclusively from train edges."""
    train_indices = graph.get_ppi_indices("train")
    train_targets = graph.ppi[train_indices]
    target_local = torch.searchsorted(full_node_index, train_targets)
    if target_local.numel() and target_local.max() >= full_node_index.numel():
        raise ValueError("training edge endpoint is absent from the full graph")
    if not torch.equal(full_node_index[target_local], train_targets):
        raise ValueError("training edge endpoint is absent from the full graph")
    return EdgeRelationLookup.from_pairs(
        target_local,
        graph.ppi_labels[train_indices],
        num_nodes=full_node_index.numel(),
    )


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    use_edge_relations = getattr(args, "use_edge_relations", False)
    use_sampler_edge_relations = getattr(
        args, "use_sampler_edge_relations", False
    )
    graph = PPIGraph(
        args.dataset, args.split, root=args.root, device=device, cache_dir=args.cache_dir
    )
    esm_dim = graph.tensor.shape[1]
    # The knowledge graph is the full dataset graph for training, validation
    # and test alike; the split only selects the target pairs (and the
    # training-node set used for test visibility grouping below).
    full_graph = graph.build_full_graph(undirected=True)
    # Features are stored as bfloat16 on disk.  Convert the graph once up
    # front instead of repeatedly casting gathered features inside the
    # sampler/predictor (a pure speed optimization: element-wise conversion
    # is order-independent, so results are bit-identical to per-step casts).
    full_graph["node_feat"] = full_graph["node_feat"].to(torch.float32)
    # Training-split node set, used only to group test pairs into BS/ES/NS.
    train_node_index = graph.build_graph("train")["node_index"]
    train_indices = graph.get_ppi_indices("train")
    val_indices = graph.get_ppi_indices("val")
    test_indices = graph.get_ppi_indices("test")
    train_targets = graph.ppi[train_indices]
    train_labels = graph.ppi_labels[train_indices]
    val_targets = graph.ppi[val_indices]
    val_labels = graph.ppi_labels[val_indices]
    test_targets = graph.ppi[test_indices]
    test_labels = graph.ppi_labels[test_indices]

    if args.sampler == "rl":
        sampler = SubgraphSampler(
            esm_dim=esm_dim,
            hidden_dim=args.hidden_dim,
            max_steps=args.max_steps,
            k_hops=args.k_hops,
            relation_dim=7 if use_sampler_edge_relations else None,
            structural_features=args.sampler_structural_features,
            base=args.sampler_base,
            policy=args.sampler_policy,
        ).to(device)
    elif args.sampler == "static":
        sampler = StaticNeighborhoodSampler(
            esm_dim=esm_dim,
            hidden_dim=args.hidden_dim,
            max_steps=args.max_steps,
            k_hops=args.k_hops,
        ).to(device)
    elif args.sampler == "heuristic":
        sampler = HeuristicSampler(
            esm_dim=esm_dim,
            hidden_dim=args.hidden_dim,
            max_steps=args.max_steps,
            k_hops=args.k_hops,
            min_size=args.random_subset_min_size,
            max_size=args.random_subset_max_size,
        ).to(device)
    else:
        sampler = RandomSubsetSampler(
            esm_dim=esm_dim,
            hidden_dim=args.hidden_dim,
            max_steps=args.max_steps,
            k_hops=args.k_hops,
            min_size=args.random_subset_min_size,
            max_size=args.random_subset_max_size,
        ).to(device)
    ppr_lookup = None
    if args.readout == "attention":
        ppr_lookup = PPRLookup(
            full_graph["edge_index"],
            num_nodes=int(full_graph["node_feat"].shape[0]),
            alpha=args.ppr_alpha,
            eps=args.ppr_eps,
        )
    predictor = PPIPredictor(
        esm_dim=esm_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.gnn_layers,
        heads=args.heads,
        dropout=args.dropout,
        edge_dim=7 if use_edge_relations else None,
        readout=args.readout,
        ppr=ppr_lookup,
    ).to(device)
    sampler_optimizer = (
        torch.optim.Adam(sampler.parameters(), lr=args.sampler_lr)
        if args.sampler == "rl" and args.sampler_policy == "learned"
        else None  # non-learnable samplers and the uniform control policy
                   # never receive a sampler update
    )
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)
    edge_relations = (
        _build_training_edge_relations(graph, full_graph["node_index"])
        if use_edge_relations or use_sampler_edge_relations else None
    )
    trainer = AlternatingTrainer(
        sampler,
        predictor,
        sampler_optimizer,
        predictor_optimizer,
        reinforce_gamma=args.reinforce_gamma,
        edge_relations=edge_relations,
        reward="margin" if args.reward_margin else "bce_diff",
        reward_pos=args.reward_pos,
        reward_neg=args.reward_neg,
        reward_ref=args.reward_ref,
    )

    # Build the full-graph adjacency once and share it across every training
    # batch, the per-epoch validation pass and the final test pass; each
    # target excludes its own edge lazily inside ``SubgraphSampler.sample``.
    full_adjacency = sampler._build_adjacency(
        full_graph["edge_index"], full_graph["node_feat"].shape[0]
    )

    results = []
    experiment_start = time.perf_counter()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    best_epoch = None
    best_val_macro_auc = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = torch.randperm(train_targets.shape[0], device=device)
        batch_records = []
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            batch_size = batch_indices.numel()
            metrics = trainer.alternating_batch_step(
                full_graph["node_feat"],
                full_graph["edge_index"],
                train_targets[batch_indices],
                train_labels[batch_indices],
                full_graph["node_index"],
                adjacency=full_adjacency,
            )
            batch_records.append((batch_size, metrics))

        train_metrics = _aggregate_train_metrics(batch_records)
        val_metrics = evaluate(
            trainer,
            full_graph["node_feat"],
            full_graph,
            val_targets,
            val_labels,
            args.eval_batch_size,
            train_node_index,
            adjacency=full_adjacency,
        )
        record = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "seconds": time.perf_counter() - epoch_start,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

        # Select the best checkpoint on validation Macro-AUC only.  A None
        # value (undefined for a constant-class / empty subset) never wins.
        val_macro_auc = val_metrics["roc_auc_macro"]
        if (
            val_macro_auc is not None
            and (best_val_macro_auc is None or val_macro_auc > best_val_macro_auc)
        ):
            best_epoch = epoch
            best_val_macro_auc = val_macro_auc
            best_state = {
                "epoch": epoch,
                "sampler": copy.deepcopy(sampler.state_dict()),
                "predictor": copy.deepcopy(predictor.state_dict()),
            }
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_payload = {
                    "epoch": epoch,
                    "sampler": sampler.state_dict(),
                    "predictor": predictor.state_dict(),
                    "predictor_optimizer": predictor_optimizer.state_dict(),
                }
                # The non-learnable samplers (static / random-subset /
                # heuristic) run without a sampler optimizer.
                if sampler_optimizer is not None:
                    checkpoint_payload["sampler_optimizer"] = (
                        sampler_optimizer.state_dict()
                    )
                torch.save(
                    checkpoint_payload, checkpoint_dir / f"best_{epoch}.pt"
                )

    # Strict protocol: evaluate the test set exactly once, after training, on
    # the best validation checkpoint. When no checkpoint directory is given,
    # use the in-memory copy captured at the best epoch.
    best_checkpoint_test = None
    if best_epoch is not None:
        checkpoint = best_state
        if checkpoint_dir is not None:
            checkpoint = torch.load(
                checkpoint_dir / f"best_{best_epoch}.pt", weights_only=True
            )
        sampler.load_state_dict(checkpoint["sampler"])
        predictor.load_state_dict(checkpoint["predictor"])
        best_checkpoint_test = {
            "epoch": best_epoch,
            **{
                f"test_{key}": value
                for key, value in evaluate(
                    trainer,
                    full_graph["node_feat"],
                    full_graph,
                    test_targets,
                    test_labels,
                    args.eval_batch_size,
                    train_node_index,
                    adjacency=full_adjacency,
                ).items()
            },
        }

    output = {
        "config": vars(args),
        "total_seconds": time.perf_counter() - experiment_start,
        "best_epoch": best_epoch,
        "best_val_macro_auc": best_val_macro_auc,
        "best_checkpoint_test": best_checkpoint_test,
        "epochs": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--cache-dir", default="dataset/.cache")
    parser.add_argument(
        "--dataset",
        choices=sorted(PPIGraph.AVAILABLE_SPLITS),
        default="SHS27k",
        help="dataset to train on; one of SHS27k, SHS148k, STRING",
    )
    parser.add_argument(
        "--split", choices=["bfs", "dfs", "random"], default="bfs",
        help="split method; the valid choices depend on --dataset "
             "(STRING only provides dfs)",
    )
    parser.add_argument(
        "--output", default=None,
        help="optional JSON path for saving experiment metrics",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="optional directory for saving the best checkpoint (selected by "
             "validation Macro-AUC) and replaying the test set on it",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--sampler", choices=["rl", "static", "random-subset", "heuristic"],
        default="rl",
        help="rl: learned REINFORCE subgraph sampler (default); static: "
             "non-learnable sampler that takes the whole safe k-hops "
             "neighborhood of G0 (ablation for the effect of RL selection, "
             "predictor-only training); random-subset: non-learnable sampler "
             "that takes a uniformly random subset of the k-hops region with "
             "an RL-sized node budget (separates selection strategy from "
             "context amount); heuristic: non-learnable sampler that takes a "
             "topology-ranked subset of the k-hops region (common target "
             "neighbors first, then by degree) with the same budget as "
             "random-subset (diagnostic for whether an informed selection "
             "rule beats random at that budget)",
    )
    parser.add_argument(
        "--random-subset-min-size", type=int, default=3,
        help="minimum final node count for --sampler random-subset/heuristic "
             "(targets are always included)",
    )
    parser.add_argument(
        "--random-subset-max-size", type=int, default=7,
        help="maximum final node count for --sampler random-subset/heuristic",
    )
    parser.add_argument(
        "--k-hops", type=int, default=1,
        help="maximum safe-adjacency distance from G0 seeds for sampler actions",
    )
    parser.add_argument(
        "--sampler-base", choices=["none", "static"], default="none",
        help="augmented RL semantics: fix the candidate region on the G0 "
             "seeds, then seed every trajectory with the full static 1-hop "
             "base (the graph --sampler static with k-hops 1 returns) before "
             "any learned action, so RL only chooses additions from the rest "
             "of the region; k-hops 1 leaves an empty frontier and "
             "degenerates to the static sampler; requires --sampler rl",
    )
    parser.add_argument(
        "--sampler-policy", choices=["learned", "uniform"], default="learned",
        help="action policy of the RL sampler: learned (default, scored by "
             "the network and trained with REINFORCE) or uniform (control "
             "arm: additions are drawn uniformly from the same frontier at "
             "train and eval time, without sampler training); requires "
             "--sampler rl",
    )
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-edge-relations", action="store_true",
        help="use 7-D relations on train edges in Predictor GAT; validation, "
             "test and virtual edges remain all-zero",
    )
    parser.add_argument(
        "--use-sampler-edge-relations", action="store_true",
        help="use train-edge 7-D relations when the RL sampler scores frontier "
             "candidates; held-out, target and virtual edges remain all-zero",
    )
    parser.add_argument(
        "--sampler-structural-features", action="store_true",
        help="add 8-D topology features (common-neighbor, target-touching, "
             "degree, selected-connectivity, target distances, Adamic-Adar) "
             "to the RL sampler's candidate scoring, with a linear skip "
             "channel initialized to the heuristic ranking (greedy starts "
             "equal to --sampler heuristic); requires --sampler rl",
    )
    parser.add_argument("--sampler-lr", type=float, default=1e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-3)
    parser.add_argument(
        "--readout", choices=["mean", "attention"], default="mean",
        help="predictor readout: mean (default, historical mean pooling) or "
             "attention (keep mean pooling and add a target-anchored "
             "LinkAttention summary with PPR positional encodings, "
             "RISE-DDI style; requires --ppr-alpha/--ppr-eps for the "
             "label-free PPR precomputation)",
    )
    parser.add_argument(
        "--ppr-alpha", type=float, default=0.15,
        help="teleport probability of the PPR random walks (--readout "
             "attention)",
    )
    parser.add_argument(
        "--ppr-eps", type=float, default=5e-6,
        help="forward-push accuracy threshold of the PPR rows (--readout "
             "attention)",
    )
    parser.add_argument("--reinforce-gamma", type=float, default=1.0)
    parser.add_argument(
        "--reward-margin", action="store_true",
        help="score sampler actions by the label-aligned mean probability "
             "margin M(p) = mean_j((2y_j-1)*p_j): each step is rewarded by "
             "its margin improvement over the fixed G0 reference, scaled by "
             "--reward-pos/--reward-neg for improvements/regressions "
             "(RISE-DDI-style), instead of the sequential BCE difference",
    )
    parser.add_argument(
        "--reward-pos", type=float, default=2.0,
        help="reward scale for margin improvements (--reward-margin)",
    )
    parser.add_argument(
        "--reward-neg", type=float, default=1.0,
        help="reward scale for margin regressions (--reward-margin)",
    )
    parser.add_argument(
        "--reward-ref", choices=["initial", "base"], default="initial",
        help="reference graph of the reward: initial (default, G0 = target "
             "pair plus virtual proxies, historical behavior) or base "
             "(marginal reward against the augmented static base prediction, "
             "RISE-DDI-style pred_default; requires --sampler rl "
             "--sampler-base static)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.split not in PPIGraph.AVAILABLE_SPLITS[args.dataset]:
        parser.error(
            f"split {args.split!r} is not available for dataset "
            f"{args.dataset!r}; expected one of "
            f"{PPIGraph.AVAILABLE_SPLITS[args.dataset]}"
        )
    split_path = Path(args.root) / f"{args.dataset}_{args.split}.json"
    if not split_path.is_file():
        parser.error(
            f"split file {split_path} does not exist; check --root together "
            f"with --dataset/--split (e.g. SHS148k bfs is only provided by "
            f"--root dataset_ppisplit)"
        )
    if args.k_hops < 0:
        parser.error("k-hops must be non-negative")
    if args.random_subset_min_size < 2:
        parser.error("random-subset-min-size must be at least 2")
    if args.random_subset_max_size < args.random_subset_min_size:
        parser.error("random-subset-max-size must be >= random-subset-min-size")
    if args.use_sampler_edge_relations and args.sampler != "rl":
        parser.error("sampler edge relations require --sampler rl")
    if args.sampler_structural_features and args.sampler != "rl":
        parser.error("sampler structural features require --sampler rl")
    if args.sampler_base != "none" and args.sampler != "rl":
        parser.error("sampler-base requires --sampler rl")
    if args.sampler_policy != "learned" and args.sampler != "rl":
        parser.error("sampler-policy requires --sampler rl")
    if args.reward_ref == "base" and (
        args.sampler != "rl" or args.sampler_base != "static"
    ):
        parser.error(
            "reward-ref base requires --sampler rl --sampler-base static"
        )
    if args.reward_pos <= 0.0 or args.reward_neg <= 0.0:
        parser.error("reward-pos and reward-neg must be positive")
    if not 0.0 < args.ppr_alpha < 1.0:
        parser.error("ppr-alpha must be in (0, 1)")
    if args.ppr_eps <= 0.0:
        parser.error("ppr-eps must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
