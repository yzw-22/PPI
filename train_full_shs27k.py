"""SHS27k 全量数据 10 轮交替训练脚本。

与 train_shs27k.py（600 样本子集采样）不同，本脚本**每轮在完整训练集**上执行
Sampler 训练步（REINFORCE，固定 Predictor）与 Predictor 训练步（BCE 监督，固定
Sampler），并在每轮之后报告：

- 训练集损失：Predictor 在**完整训练集**上的平均 BCE（评估模式，Sampler argmax）
- 验证集指标：用于在验证集上搜索使 macro-F1 最大的决策阈值
- 测试集性能：用验证集调优后的阈值评估**完整测试集**上的 micro/macro AUC 与 F1

结果输出到 ``results/full_train_shs27k_10rounds.json``。
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


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

NUM_ROUNDS = 10
DATASET = "SHS27k"
SPLIT = "dataset/SHS27k_bfs.json"
OUT_DIR = Path("results")
SEED = 42

config = PPIConfig(
    T_max=10,
    gnn_num_layers=3,
    gnn_dropout=0.3,
    gnn_heads=4,
    hidden_dim=256,
    attention_dim=64,
    lr_sampler=1e-4,
    lr_predictor=1e-3,
    sampler_steps=1,
    predictor_steps=1,
    reinforce_baseline_coef=0.5,
)


# ---------------------------------------------------------------------------
# 评估工具（基于预计算预测矩阵，避免每轮重复采样）
# ---------------------------------------------------------------------------


def metrics_from_matrix(y_pred, y_true, threshold):
    """由预测矩阵计算多标签 AUC / F1（与 PPIModel.evaluate 同口径）。"""
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
    valid_auc = [a for a in auc_per_label if not np.isnan(a)]
    auc_macro = float(np.mean(valid_auc)) if valid_auc else float("nan")

    y_pred_bin = (y_pred >= threshold).astype(np.int32)
    f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    return {
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def tune_threshold_on_matrix(y_pred, y_true, num_candidates=100):
    """在 [0.01, 0.99] 上搜索使 macro-F1 最大的全局阈值。"""
    thresholds = np.linspace(0.01, 0.99, num_candidates)
    best_th, best_f1 = 0.5, -1.0
    for th in thresholds:
        f1 = f1_score(y_true, (y_pred >= th).astype(np.int32),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return float(best_th)


@torch.no_grad()
def train_bce(model, ppi_indices):
    """计算 Predictor 在给定 PPI 集上的平均 BCE（评估模式，Sampler argmax）。"""
    y_pred, y_true = model._predict_matrix(ppi_indices)
    preds = torch.tensor(y_pred, dtype=torch.float32)
    labels = torch.tensor(y_true, dtype=torch.float32)
    return F.binary_cross_entropy(preds, labels).item()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"[Experiment] {DATASET} 全量数据训练, Split=bfs, Rounds={NUM_ROUNDS}")
    print(f"[Experiment] Device={config.device}")
    print()

    model = PPIModel.from_dataset(config, DATASET, "dataset", verbose=False)
    model.set_esm_tensor(f"dataset/{DATASET}_tensor.pt")
    split = model.load_split(SPLIT)

    train_idx = split["train_index"]   # 全量训练集
    val_idx = split["val_index"]       # 全量验证集
    test_idx = split["test_index"]     # 全量测试集

    print(f"[Experiment] train={len(train_idx)}, val={len(val_idx)}, "
          f"test={len(test_idx)}")

    history = []
    header = (f"{'Rnd':>3} | {'TrainBCE':>8} | {'SamReward':>9} | "
              f"{'ValAUC':>6} | {'Thr':>5} | {'TestAUC':>7} | "
              f"{'TestF1mac':>9} | {'TestF1mic':>9} | {'Time':>6}")
    print(header)
    print("-" * len(header))

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()

        # ---- 交替训练（各一个全量步） ----
        sampler_info = model.train_sampler_step(train_idx)
        predictor_info = model.train_predictor_step(train_idx)

        # ---- 训练集损失（更新后的模型，评估模式） ----
        tr_bce = train_bce(model, train_idx)

        # ---- 验证集：预计算矩阵 → 调阈值 + 指标 ----
        val_pred, val_true = model._predict_matrix(val_idx)
        best_th = tune_threshold_on_matrix(val_pred, val_true)
        val_metrics = metrics_from_matrix(val_pred, val_true, best_th)

        # ---- 测试集：用验证集阈值评估 ----
        test_pred, test_true = model._predict_matrix(test_idx)
        test_metrics = metrics_from_matrix(test_pred, test_true, best_th)
        test_metrics_05 = metrics_from_matrix(test_pred, test_true, 0.5)

        elapsed = time.time() - t0
        rec = {
            "round": rnd,
            "train_bce": tr_bce,
            "sampler_reward": sampler_info["reward"],
            "policy_loss": sampler_info["policy_loss"],
            "value_loss": sampler_info["value_loss"],
            "predictor_bce_in_step": predictor_info["bce_loss"],
            "best_threshold": best_th,
            "val_auc_micro": val_metrics["auc_micro"],
            "val_auc_macro": val_metrics["auc_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "test_auc_micro": test_metrics["auc_micro"],
            "test_auc_macro": test_metrics["auc_macro"],
            "test_f1_macro_best": test_metrics["f1_macro"],
            "test_f1_micro_best": test_metrics["f1_micro"],
            "test_f1_macro_05": test_metrics_05["f1_macro"],
            "test_f1_micro_05": test_metrics_05["f1_micro"],
            "elapsed_s": elapsed,
        }
        history.append(rec)

        print(
            f"{rnd:>3} | {tr_bce:>8.4f} | {rec['sampler_reward']:>9.4f} | "
            f"{rec['val_auc_micro']:>6.4f} | {best_th:>5.2f} | "
            f"{rec['test_auc_micro']:>7.4f} | {rec['test_f1_macro_best']:>9.4f} | "
            f"{rec['test_f1_micro_best']:>9.4f} | {elapsed:>5.0f}s"
        )

    # 保存结果
    out_path = OUT_DIR / "full_train_shs27k_10rounds.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2,
                  default=lambda x: float(x) if np.isscalar(x) else x)
    print(f"\n[Experiment] Results saved to {out_path}")

    # 汇总趋势
    print("\n[Experiment] Summary (round 1 vs round 10):")
    first, last = history[0], history[-1]
    for key in ["train_bce", "sampler_reward", "test_auc_micro",
                "test_auc_macro", "test_f1_macro_best"]:
        print(f"  {key:>22}: {first[key]:.4f}  →  {last[key]:.4f}")


if __name__ == "__main__":
    main()
