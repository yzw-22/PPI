"""独立可视化脚本：用训练产出的最佳检查点提取并绘制 SHS27k BFS 最终子图。

不修改 src/ 训练代码。用法：

    python tools/vis_subgraphs.py \
      --json /tmp/ppi_shs27k_bfs_vis.json \
      --checkpoint-dir /tmp/ppi_ckpt_vis \
      --device cpu --out-dir figures

从 --json 的 config 读取超参构造模型；从 --checkpoint-dir 加载最佳检查点；
在测试集上挑选有/无 proxy 的 PPI 样例，贪婪采样后绘制 final_graph。

proxy 说明：main 分支的 SamplingTrajectory 不暴露 proxy 节点，脚本按
sampler._add_virtual_proxies 的同一逻辑（孤立端点 + 余弦最近非目标节点）
确定性重算，用于标注；虚拟边 = 孤立端点与其 proxy 之间的边。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.ppi_graph import PPIGraph
from src.predictor import PPIPredictor
from src.sampler import SubgraphSampler
from src.trainer import AlternatingTrainer


ROLE_COLORS = {
    "u": "#d62728",
    "v": "#1f77b4",
    "proxy": "#9467bd",
    "context": "#9ecae1",
}
ROLE_SHAPES = {
    "u": "s",
    "v": "s",
    "proxy": "D",
    "context": "o",
}
ROLE_LABELS = {"u": "u", "v": "v", "proxy": "proxy", "context": ""}


def classify_target(adjacency, u_local, v_local):
    """返回 (u_isolated, v_isolated)：端点安全邻接是否为空。"""
    return (not adjacency[u_local] - {v_local},
            not adjacency[v_local] - {u_local})


def recompute_proxies(adjacency, node_features, u_local, v_local):
    """镜像 _add_virtual_proxies：返回 {局部端点: proxy 局部 id}。

    注意：sampler 内部使用"安全邻接"（目标边已双向移除），因此这里的
    孤立判定也必须排除目标对端，否则会漏判。
    """
    available = torch.ones(node_features.shape[0], dtype=torch.bool)
    available[[u_local, v_local]] = False
    proxies = {}
    for node, partner in ((u_local, v_local), (v_local, u_local)):
        if (adjacency[node] - {partner}) or not available.any():
            continue
        node_vec = torch.nn.functional.normalize(node_features[node].float(), dim=0)
        candidates = torch.nn.functional.normalize(node_features.float(), dim=1)
        scores = candidates @ node_vec
        scores = scores.masked_fill(~available, -torch.inf)
        proxies[node] = int(scores.argmax())
    return proxies


def collect_samples(graph, split_graph, targets, adjacency, sampler,
                    node_features, node_index, n_total=6):
    """按 proxy 场景挑选样例并采样，返回样例元信息列表。"""
    node_index_np = node_index.tolist()
    buckets = {"both": [], "one": [], "none": []}
    for idx, t in enumerate(targets):
        u_g, v_g = int(t[0]), int(t[1])
        u_l = node_index_np.index(u_g)
        v_l = node_index_np.index(v_g)
        u_iso, v_iso = classify_target(adjacency, u_l, v_l)
        key = "both" if (u_iso and v_iso) else ("one" if (u_iso or v_iso) else "none")
        buckets[key].append((idx, t.tolist(), u_l, v_l, u_iso, v_iso))

    # 固定配比保证覆盖三种场景：双孤立(2 proxy) / 单端孤立(1 proxy) / 无 proxy
    plan = [("both", 1), ("one", 2), ("none", 3)]
    picked = []
    for key, want in plan:
        picked.extend(buckets[key][:want])
    for key in ("both", "one", "none"):  # 不足时用其余场景补齐
        for item in buckets[key]:
            if len(picked) >= n_total:
                break
            if item not in picked:
                picked.append(item)
    if len(picked) < n_total:
        raise SystemExit(f"样例不足：test 只有 {len(picked)} 个可选")

    samples = []
    for idx, target, u_l, v_l, u_iso, v_iso in picked:
        trajectory = sampler.sample(
            node_features, split_graph["edge_index"], torch.tensor(target),
            node_index, training=False, adjacency=adjacency,
        )
        final = trajectory.final_graph
        proxies = recompute_proxies(adjacency, node_features, u_l, v_l)
        proxy_global = {int(node_index[p]) for p in proxies.values()}
        samples.append({
            "index": int(idx),
            "target": target,
            "u_isolated": u_iso,
            "v_isolated": v_iso,
            "proxy_count": len(proxies),
            "proxy_local": {int(k): int(v) for k, v in proxies.items()},
            "steps": len(trajectory.steps),
            "final_node_index": final.node_index.tolist(),
            "final_edge_index": final.edge_index.tolist(),
            "final_target_nodes": final.target_nodes.tolist(),
            "final_global_nodes": final.node_index.tolist(),
        })
    return samples


def draw_panel(ax, sample, node_index, label_rows):
    """绘制单个 final_graph 面板。"""
    global_ids = sample["final_global_nodes"]
    u_g, v_g = sample["target"]
    # proxy_local 是 split 局部 id（feature 行号），必须经完整 node_index 映射
    proxy_locals = set(sample["proxy_local"].values())
    proxy_globals = {int(node_index[i]) for i in proxy_locals}
    virtual_pairs = set()
    for node_l, proxy_l in sample["proxy_local"].items():
        node_l, proxy_l = int(node_l), int(proxy_l)
        virtual_pairs.add(tuple(sorted((int(node_index[node_l]),
                                        int(node_index[proxy_l])))))

    G = nx.Graph()
    G.add_nodes_from(global_ids)
    edge_pairs = set()
    for s, t in zip(*sample["final_edge_index"]):
        edge_pairs.add(tuple(sorted((global_ids[s], global_ids[t]))))
    G.add_edges_from(edge_pairs)

    # 布局：u/v/proxy 锚定，其余节点弹簧布局
    fixed = {u_g: (-1.6, 0.0), v_g: (1.6, 0.0)}
    for i, g in enumerate(proxy_globals):
        fixed[g] = (0.0, -1.8 - 0.8 * i)
    pos = nx.spring_layout(G, pos=fixed, fixed=list(fixed),
                           seed=42, k=1.1, iterations=300)

    roles = {}
    for g in global_ids:
        if g == u_g:
            roles[g] = "u"
        elif g == v_g:
            roles[g] = "v"
        elif g in proxy_globals:
            roles[g] = "proxy"
        else:
            roles[g] = "context"

    for role in ("u", "v", "proxy", "context"):
        nodes = [g for g, r in roles.items() if r == role]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=ROLE_COLORS[role],
            node_shape=ROLE_SHAPES[role], node_size=420 if role != "context" else 260,
            ax=ax, edgecolors="black", linewidths=1.0,
        )
    real_edges = [e for e in edge_pairs if e not in virtual_pairs]
    if real_edges:
        nx.draw_networkx_edges(G, pos, edgelist=real_edges, ax=ax,
                               edge_color="#888888", width=1.2)
    if virtual_pairs:
        nx.draw_networkx_edges(G, pos, edgelist=sorted(virtual_pairs), ax=ax,
                               edge_color="#d62728", width=1.8, style="dashed")
    labels = {g: (ROLE_LABELS[roles[g]] or str(g)) for g in global_ids}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    proxy_txt = "proxy" if sample["proxy_count"] else "no proxy"
    u_iso = "u孤立" if sample["u_isolated"] else ""
    v_iso = "v孤立" if sample["v_isolated"] else ""
    iso_txt = ",".join(filter(None, [u_iso, v_iso]))
    ax.set_title(
        f"PPI ({u_g},{v_g}) | {proxy_txt} | {iso_txt}\n"
        f"nodes={len(global_ids)} edges={len(edge_pairs)} steps={sample['steps']}",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="训练输出 JSON")
    parser.add_argument("--checkpoint-dir", required=True, help="训练 checkpoint 目录")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--n-samples", type=int, default=6)
    args = parser.parse_args()

    result = json.loads(Path(args.json).read_text())
    cfg = result["config"]
    best_epoch = result["best_epoch"]
    print(f"best_epoch={best_epoch} (val macro AUC={result['best_val_macro_auc']:.4f})")

    device = torch.device(args.device)
    graph = PPIGraph(cfg["dataset"], cfg["split"], root=cfg["root"], device=device)
    esm_dim = graph.tensor.shape[1]
    test_graph = graph.build_graph("test")
    test_targets = graph.ppi[graph.get_ppi_indices("test")]
    node_index = test_graph["node_index"]
    node_features = test_graph["node_feat"]
    adjacency = SubgraphSampler._build_adjacency(
        test_graph["edge_index"], node_features.shape[0])

    sampler = SubgraphSampler(
        esm_dim=esm_dim, hidden_dim=cfg["hidden_dim"],
        max_steps=cfg["max_steps"], k_hops=cfg["k_hops"],
    ).to(device)
    predictor = PPIPredictor(
        esm_dim=esm_dim, hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["gnn_layers"], heads=cfg["heads"], dropout=cfg["dropout"],
    ).to(device)
    ckpt = torch.load(Path(args.checkpoint_dir) / f"best_{best_epoch}.pt",
                      weights_only=True)
    sampler.load_state_dict(ckpt["sampler"])
    predictor.load_state_dict(ckpt["predictor"])
    sampler.eval()
    predictor.eval()
    trainer = AlternatingTrainer(sampler, predictor, None, None)

    samples = collect_samples(
        graph, test_graph, test_targets, adjacency, sampler,
        node_features, node_index, n_total=args.n_samples,
    )

    n_cols = 3
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    for i, sample in enumerate(samples):
        ax = axes.flat[i]
        draw_panel(ax, sample, node_index, [])
    for j in range(len(samples), n_rows * n_cols):
        axes.flat[j].axis("off")

    handles = [
        plt.Line2D([], [], marker="s", color=ROLE_COLORS["u"], linestyle="None",
                   markersize=9, label="目标 u"),
        plt.Line2D([], [], marker="s", color=ROLE_COLORS["v"], linestyle="None",
                   markersize=9, label="目标 v"),
        plt.Line2D([], [], marker="D", color=ROLE_COLORS["proxy"], linestyle="None",
                   markersize=9, label="虚拟 proxy"),
        plt.Line2D([], [], marker="o", color=ROLE_COLORS["context"], linestyle="None",
                   markersize=8, label="上下文节点"),
        plt.Line2D([], [], color="#d62728", linestyle="--", linewidth=2,
                   label="虚拟 proxy 边"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"SHS27k {cfg['split']} 最终子图（best epoch {best_epoch}，"
                 f"k_hops={cfg['k_hops']}，max_steps={cfg['max_steps']}）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "shs27k_bfs_final_subgraphs.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"图已保存: {png_path}")
    meta_path = out_dir / "shs27k_bfs_final_subgraphs.json"
    meta_path.write_text(json.dumps(samples, indent=2) + "\n")
    print(f"元数据已保存: {meta_path}")
    for s in samples:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
