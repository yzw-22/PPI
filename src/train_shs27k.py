"""Run an alternating-training PPI experiment.

Supports the SHS27k / SHS148k / STRING datasets via ``--dataset`` and their
available bfs / dfs / random splits via ``--split`` (the valid split choices
depend on the dataset; see ``PPIGraph.AVAILABLE_SPLITS``). Dataset files are
assumed to exist under ``--root`` and are not pre-checked.
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
from .predictor import PPIPredictor
from .sampler import SubgraphSampler
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


def evaluate(trainer, node_features, split_graph, targets, labels, batch_size,
             training_node_index):
    """Evaluate final graphs and group pairs by training-node visibility."""
    trainer.sampler.eval()
    trainer.predictor.eval()
    adjacency = trainer.sampler._build_adjacency(
        split_graph["edge_index"], node_features.shape[0]
    )
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            target_batch = targets[start:start + batch_size]
            trajectories = [
                trainer.sampler.sample(
                    node_features,
                    split_graph["edge_index"],
                    target,
                    split_graph["node_index"],
                    training=False,
                    adjacency=adjacency,
                )
                for target in target_batch
            ]
            graphs = [trajectory.final_graph for trajectory in trajectories]
            logits = trainer._predict_graphs(node_features, graphs)
            probabilities.append(torch.sigmoid(logits).cpu())

    probabilities = torch.cat(probabilities).numpy()
    labels = labels.cpu().numpy()
    metrics = _metrics(labels, probabilities)

    training_nodes = set(training_node_index.detach().cpu().tolist())
    groups = {"BS": [], "ES": [], "NS": []}
    for sample_index, (source, target) in enumerate(targets.detach().cpu().tolist()):
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
    sample_count = 0
    for batch_size, metrics in batch_records:
        steps = int(metrics["sampler_step_count"])
        sampler_loss_total += float(metrics["sampler_loss"]) * steps
        reward_total += float(metrics["mean_reward"]) * steps
        sampler_step_count += steps

        sampler_size = int(batch_size)
        predictor_loss_total += float(metrics["predictor_loss"]) * sampler_size
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
    }


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    graph = PPIGraph(
        args.dataset, args.split, root=args.root, device=device, cache_dir=args.cache_dir
    )
    esm_dim = graph.tensor.shape[1]
    train_graph = graph.build_graph("train", undirected=True)
    val_graph = graph.build_graph("val", undirected=True)
    test_graph = graph.build_graph("test", undirected=True)
    # Features are stored as bfloat16 on disk.  Convert each split once up
    # front instead of repeatedly casting gathered features inside the
    # sampler/predictor (a pure speed optimization: element-wise conversion
    # is order-independent, so results are bit-identical to per-step casts).
    for split_graph in (train_graph, val_graph, test_graph):
        split_graph["node_feat"] = split_graph["node_feat"].to(torch.float32)
    train_indices = graph.get_ppi_indices("train")
    val_indices = graph.get_ppi_indices("val")
    test_indices = graph.get_ppi_indices("test")
    train_targets = graph.ppi[train_indices]
    train_labels = graph.ppi_labels[train_indices]
    val_targets = graph.ppi[val_indices]
    val_labels = graph.ppi_labels[val_indices]
    test_targets = graph.ppi[test_indices]
    test_labels = graph.ppi_labels[test_indices]

    sampler = SubgraphSampler(
        esm_dim=esm_dim,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        k_hops=args.k_hops,
    ).to(device)
    predictor = PPIPredictor(
        esm_dim=esm_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.gnn_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr=args.sampler_lr)
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)
    trainer = AlternatingTrainer(
        sampler,
        predictor,
        sampler_optimizer,
        predictor_optimizer,
        reinforce_gamma=args.reinforce_gamma,
    )

    # Build the split-level adjacency once instead of once per batch; each
    # target excludes its own edge lazily inside ``SubgraphSampler.sample``.
    train_adjacency = sampler._build_adjacency(
        train_graph["edge_index"], train_graph["node_feat"].shape[0]
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
                train_graph["node_feat"],
                train_graph["edge_index"],
                train_targets[batch_indices],
                train_labels[batch_indices],
                train_graph["node_index"],
                adjacency=train_adjacency,
            )
            batch_records.append((batch_size, metrics))

        train_metrics = _aggregate_train_metrics(batch_records)
        val_metrics = evaluate(
            trainer,
            val_graph["node_feat"],
            val_graph,
            val_targets,
            val_labels,
            args.eval_batch_size,
            train_graph["node_index"],
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
                torch.save(
                    {
                        "epoch": epoch,
                        "sampler": sampler.state_dict(),
                        "predictor": predictor.state_dict(),
                        "sampler_optimizer": sampler_optimizer.state_dict(),
                        "predictor_optimizer": predictor_optimizer.state_dict(),
                    },
                    checkpoint_dir / f"best_{epoch}.pt",
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
                    test_graph["node_feat"],
                    test_graph,
                    test_targets,
                    test_labels,
                    args.eval_batch_size,
                    train_graph["node_index"],
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
        "--k-hops", type=int, default=1,
        help="maximum safe-adjacency distance from G0 seeds for sampler actions",
    )
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sampler-lr", type=float, default=1e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-3)
    parser.add_argument("--reinforce-gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.split not in PPIGraph.AVAILABLE_SPLITS[args.dataset]:
        parser.error(
            f"split {args.split!r} is not available for dataset "
            f"{args.dataset!r}; expected one of "
            f"{PPIGraph.AVAILABLE_SPLITS[args.dataset]}"
        )
    if args.k_hops < 0:
        parser.error("k-hops must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
