"""Predictor 消融实验：残差稀释 + 容量（hidden_dim / gnn_num_layers）。

动机：全量数据训练中 Sampler 奖励 ≈ 0，初步探针显示"子图改变 → pairwise readout
变化 60%，但输出 logits 仅变化 1.4%"。本脚本在固定 600 样本子集上训练多个 Predictor
变体（各 6 轮），对比完整测试集性能，并集成诊断：

  1. 结构扰动敏感性：G_0={u,v} / 采样子图 / 随机邻居子图 → 输出 logits 的相对变化
  2. 消息传递消融：完整边 vs 仅自环（去掉邻居聚合）→ pairwise / logits 是否变化
  3. 残差稀释：逐层 ||消息传递输出|| / ||残差输入|| 之比

变体：
  - baseline      隐藏 256 / 3 层 / 残差 scale=1（当前默认）
  - hidden512     隐藏 512 / 3 层
  - layers5       隐藏 256 / 5 层
  - residual_x2   残差 scale=2（放大消息传递项）
  - no_residual   去掉残差连接（h = h_new）
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from model.config import PPIConfig
from model.ppi_model import PPIModel


SEED = 42
NUM_ROUNDS = 6
SAMPLES = 600          # 每轮训练样本数（子集，控制运行时长）
PROBE_PPIS = 40        # 诊断用的 PPI 数量
DATASET = "SHS27k"
SPLIT = "dataset/SHS27k_bfs.json"
OUT_DIR = Path("results")
ESM = f"dataset/{DATASET}_tensor.pt"

VARIANTS = [
    ("baseline", dict()),
    ("hidden512", dict(hidden_dim=512)),
    ("layers5", dict(gnn_num_layers=5)),
    ("residual_x2", dict(gnn_residual_scale=2.0)),
    ("no_residual", dict(gnn_residual=False)),
]


def make_config(**overrides):
    base = dict(
        T_max=10, gnn_num_layers=3, gnn_dropout=0.3, gnn_heads=4,
        hidden_dim=256, attention_dim=64,
        lr_sampler=1e-4, lr_predictor=1e-3,
        sampler_steps=1, predictor_steps=1, reinforce_baseline_coef=0.5,
    )
    base.update(overrides)
    return PPIConfig(**base)


# ---------------------------------------------------------------------------
# 评估（与 train_full_shs27k.py 同口径）
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_matrix(model, ppi_indices):
    y_pred, y_true = model._predict_matrix(ppi_indices)
    return y_pred, y_true


def metrics(y_pred, y_true, threshold):
    auc_per_label = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
            auc_per_label.append(float("nan"))
        else:
            auc_per_label.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    try:
        auc_micro = roc_auc_score(y_true.ravel(), y_pred.ravel())
    except ValueError:
        auc_micro = float("nan")
    valid = [a for a in auc_per_label if not np.isnan(a)]
    auc_macro = float(np.mean(valid)) if valid else float("nan")
    yb = (y_pred >= threshold).astype(np.int32)
    f1_macro = f1_score(y_true, yb, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, yb, average="micro", zero_division=0)
    return {"auc_micro": auc_micro, "auc_macro": auc_macro,
            "f1_macro": f1_macro, "f1_micro": f1_micro}


def tune_threshold(y_pred, y_true, num_candidates=100):
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.01, 0.99, num_candidates):
        f1 = f1_score(y_true, (y_pred >= th).astype(np.int32),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return float(best_th)


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_variant(model, esm, sub, core, edges, self_loop_only=False):
    """按 predictor 的逐层逻辑前向，返回 (logits, pairwise, per_layer_ratios)。

    self_loop_only=True 时仅保留自环边（去掉邻居聚合，消息 = 自身变换）。
    """
    predictor = model.predictor
    esm = esm.float()  # bfloat16 → float32（与 predictor.forward 一致）
    x, ei, ul, vl = predictor._build_subgraph(
        esm, model.graph, sub, core, edges=edges)

    if self_loop_only:
        n = x.shape[0]
        rows = torch.arange(n, device=ei.device)
        ei = torch.stack([rows, rows])  # 仅自环（GAT 自带 add_self_loops 会补全）

    h = predictor.node_proj(x)
    ratios = []
    for conv, norm in zip(predictor.convs, predictor.norms):
        h_new = conv(h, ei)
        h_new = norm(h_new)
        h_new = F.relu(h_new)
        ratios.append(h_new.norm().item() / max(h.norm().item(), 1e-8))
        if model.config.gnn_residual:
            h = h + model.config.gnn_residual_scale * h_new
        else:
            h = h_new

    hu, hv = h[ul], h[vl]
    pairwise = torch.cat([hu, hv, hu * hv, torch.abs(hu - hv)], dim=-1)
    logits = predictor.classifier(pairwise)
    return torch.sigmoid(logits), pairwise, ratios


@torch.no_grad()
def diagnose(model, probe_ppis, rng):
    """结构扰动 + 消息传递消融 + 残差稀释 诊断。"""
    sampler = model.sampler
    sampler.eval(); model.predictor.eval()
    esm = model._esm_tensor

    d_g0_gt, d_g0_gr, d_gt_gr, l_diff_gt, l_diff_gr = [], [], [], [], []
    d_pair_ablate, d_logit_ablate = [], []
    layer_ratios = [[] for _ in range(model.config.gnn_num_layers)]

    def rel(a, b):
        return (a - b).norm().item() / max(a.norm().item(), 1e-8)

    for ppi in probe_ppis:
        u, v = model.ppi_list[ppi]
        core = (u, v)
        label = model._get_label(ppi).to(esm.device)

        # G_0 = {u, v}
        p0, _, _ = _forward_variant(model, esm, [u, v], core, None)
        l0 = F.binary_cross_entropy(p0, label)

        # G_t = 采样子图
        traj = sampler(esm, model.graph, u, v, training=False)
        pt, pw_full, _ = _forward_variant(
            model, esm, traj.final_subgraph, core, traj.final_edges)
        lt = F.binary_cross_entropy(pt, label)

        # G_rand = 随机邻居子图（与 G_t 同规模）
        k = len(traj.final_subgraph) - 2
        frontier = [n for n in model.graph.get_frontier({u, v}) if n not in (u, v)]
        if k > 0 and frontier:
            k = min(k, len(frontier))
            sub_rand = [u, v] + rng.sample(frontier, k)
            pr, _, _ = _forward_variant(model, esm, sub_rand, core, None)
        else:
            pr = p0
        lr = F.binary_cross_entropy(pr, label)

        d_g0_gt.append(rel(p0, pt))
        d_g0_gr.append(rel(p0, pr))
        d_gt_gr.append(rel(pt, pr))
        l_diff_gt.append((l0 - lt).item())
        l_diff_gr.append((l0 - lr).item())

        # 消息传递消融：完整边 vs 仅自环（邻居聚合的影响）
        p_self, pw_self, _ = _forward_variant(
            model, esm, traj.final_subgraph, core, traj.final_edges,
            self_loop_only=True)
        d_pair_ablate.append(rel(pw_full, pw_self))
        d_logit_ablate.append(rel(pt, p_self))

        # 残差稀释：逐层 ||消息传递输出|| / ||残差输入||
        _, _, ratios_full = _forward_variant(
            model, esm, traj.final_subgraph, core, traj.final_edges)
        for li, r in enumerate(ratios_full):
            layer_ratios[li].append(r)

    n = len(probe_ppis)
    return {
        "d_logits_G0_vs_Gt": np.mean(d_g0_gt),
        "d_logits_G0_vs_Grand": np.mean(d_g0_gr),
        "d_logits_Gt_vs_Grand": np.mean(d_gt_gr),
        "l0_minus_lt": np.mean(l_diff_gt),
        "l0_minus_lrand": np.mean(l_diff_gr),
        "d_pairwise_self_ablate": np.mean(d_pair_ablate),
        "d_logits_self_ablate": np.mean(d_logit_ablate),
        "layer_residual_ratios": [float(np.mean(r)) for r in layer_ratios],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    # 固定训练池与评估子集（跨变体共享，保证公平比较）
    model = PPIModel.from_dataset(make_config(), DATASET, "dataset", verbose=False)
    model.set_esm_tensor(ESM)
    split = model.load_split(SPLIT)
    train_idx, val_idx, test_idx = (split["train_index"],
                                    split["val_index"], split["test_index"])
    rng = random.Random(SEED)
    pool = rng.sample(train_idx, SAMPLES)
    probe_ppis = rng.sample(train_idx, PROBE_PPIS)

    print(f"[ablate] {len(VARIANTS)} variants, {NUM_ROUNDS} rounds, "
          f"{SAMPLES} samples/round, device={model.config.device}")

    out = OUT_DIR / "ablate_predictor_results.json"
    # 断点续跑：已有变体直接跳过
    results = {}
    if out.exists():
        results = json.loads(out.read_text())
        print(f"[ablate] resuming: already have {list(results.keys())}")

    for name, overrides in VARIANTS:
        if name in results:
            print(f"[ablate] skip {name} (already trained)")
            continue
        t0 = time.time()
        cfg = make_config(**overrides)
        m = PPIModel.from_dataset(cfg, DATASET, "dataset", verbose=False)
        m.set_esm_tensor(ESM)
        print(f"\n===== Variant: {name} ({cfg.hidden_dim}h/{cfg.gnn_num_layers}l, "
              f"residual={cfg.gnn_residual}x{cfg.gnn_residual_scale}) =====")

        hist = []
        for rnd in range(1, NUM_ROUNDS + 1):
            m.train_sampler_step(pool)
            m.train_predictor_step(pool)
            vp, vt = eval_matrix(m, val_idx)
            th = tune_threshold(vp, vt)
            tp, tt = eval_matrix(m, test_idx)
            tm = metrics(tp, tt, th)
            hist.append({"round": rnd, "test_auc_micro": tm["auc_micro"],
                         "test_auc_macro": tm["auc_macro"],
                         "test_f1_macro": tm["f1_macro"],
                         "test_f1_micro": tm["f1_micro"],
                         "threshold": th})
            print(f"  r{rnd:>2}: TestAUC={tm['auc_micro']:.4f} "
                  f"F1macro={tm['f1_macro']:.4f}")

        diag = diagnose(m, probe_ppis, random.Random(SEED))
        results[name] = {"history": hist, "diagnosis": diag}
        print(f"  [diag] d_logits G0vsGt={diag['d_logits_G0_vs_Gt']:.4f} "
              f"G0vsGrand={diag['d_logits_G0_vs_Grand']:.4f} "
              f"GtvsGrand={diag['d_logits_Gt_vs_Grand']:.4f}")
        print(f"  [diag] self-ablate pairwise={diag['d_pairwise_self_ablate']:.4f} "
              f"logits={diag['d_logits_self_ablate']:.4f}")
        print(f"  [diag] layer residual ratios="
              + " ".join(f"{r:.3f}" for r in diag["layer_residual_ratios"]))
        print(f"  [{time.time()-t0:.0f}s]")

        # 每个变体完成后立即保存（增量）
        with open(out, "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda x: float(x) if np.isscalar(x) else x)

    print(f"\n[ablate] saved to {out}")

    # 汇总表
    print("\n===== Summary (final round) =====")
    print(f"{'variant':>14} | {'TestAUCmi':>8} | {'TestAUCma':>8} | "
          f"{'F1macro':>8} | dG0-Gt | dSelfAbl")
    for name in results:
        last = results[name]["history"][-1]
        d = results[name]["diagnosis"]
        print(f"{name:>14} | {last['test_auc_micro']:>8.4f} | "
              f"{last['test_auc_macro']:>8.4f} | {last['test_f1_macro']:>8.4f} | "
              f"{d['d_logits_G0_vs_Gt']:>6.4f} | {d['d_logits_self_ablate']:>7.4f}")


if __name__ == "__main__":
    main()
