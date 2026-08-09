"""Run an SHS27k/bfs alternating-training experiment."""

import argparse
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
    """Return multi-label metrics for one evaluation subset."""
    if len(labels) == 0:
        return {
            "count": 0,
            "roc_auc_macro": None,
            "roc_auc_micro": None,
            "f1_macro": None,
            "f1_micro": None,
        }
    predictions = probabilities >= 0.5
    return {
        "count": int(len(labels)),
        "roc_auc_macro": float(roc_auc_score(labels, probabilities, average="macro")),
        "roc_auc_micro": float(roc_auc_score(labels, probabilities, average="micro")),
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


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    graph = PPIGraph(
        "SHS27k", "bfs", root=args.root, device=device, cache_dir=args.cache_dir
    )
    train_graph = graph.build_graph("train", undirected=True)
    val_graph = graph.build_graph("val", undirected=True)
    test_graph = graph.build_graph("test", undirected=True)
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
        esm_dim=2560,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        k_hops=args.k_hops,
    ).to(device)
    predictor = PPIPredictor(
        esm_dim=2560,
        hidden_dim=args.hidden_dim,
        num_layers=args.gnn_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    trainer = AlternatingTrainer(
        sampler,
        predictor,
        torch.optim.Adam(sampler.parameters(), lr=args.sampler_lr),
        torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr),
        reinforce_baseline_coef=args.reinforce_baseline_coef,
        reinforce_gamma=args.reinforce_gamma,
    )

    results = []
    experiment_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = torch.randperm(train_targets.shape[0], device=device)
        totals = {"sampler_loss": 0.0, "predictor_loss": 0.0, "mean_reward": 0.0}
        seen = 0
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            batch_size = batch_indices.numel()
            metrics = trainer.alternating_batch_step(
                train_graph["node_feat"],
                train_graph["edge_index"],
                train_targets[batch_indices],
                train_labels[batch_indices],
                train_graph["node_index"],
            )
            seen += batch_size
            for key in totals:
                totals[key] += float(metrics[key]) * batch_size

        train_metrics = {key: value / seen for key, value in totals.items()}
        val_metrics = evaluate(
            trainer,
            val_graph["node_feat"],
            val_graph,
            val_targets,
            val_labels,
            args.eval_batch_size,
            train_graph["node_index"],
        )
        test_metrics = evaluate(
            trainer,
            test_graph["node_feat"],
            test_graph,
            test_targets,
            test_labels,
            args.eval_batch_size,
            train_graph["node_index"],
        )
        record = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "seconds": time.perf_counter() - epoch_start,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    output = {
        "config": vars(args),
        "total_seconds": time.perf_counter() - experiment_start,
        "epochs": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--cache-dir", default="dataset/.cache")
    parser.add_argument(
        "--output", default=None,
        help="optional JSON path for saving experiment metrics",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--k-hops", type=int, default=3)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sampler-lr", type=float, default=1e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-3)
    parser.add_argument("--reinforce-baseline-coef", type=float, default=0.1)
    parser.add_argument("--reinforce-gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
