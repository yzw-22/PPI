"""SHS27k 交替训练实验脚本。

对 SHS27k（bfs split）执行 20 轮交替训练：
每轮依次执行 Sampler 训练步（REINFORCE，固定 Predictor）和
Predictor 训练步（BCE 监督，固定 Sampler）。

记录每轮：
- Sampler reward（= l_0 - l_t）
- Predictor 训练 BCE loss（训练集 loss）
- 验证集 / 测试集 AUC 与 F1

结果输出到 ``results/shs27k_20rounds.json`` 并打印汇总表。
"""

import json
import random
import time
from pathlib import Path

import numpy as np

from model.config import PPIConfig
from model.ppi_model import PPIModel


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 训练规模（subset，控制运行时长）
TRAIN_POOL = 1200          # 每轮从该池中随机抽取样本
SAMPLER_SAMPLES = 600      # 每轮 Sampler 步使用的样本数
PREDICTOR_SAMPLES = 600    # 每轮 Predictor 步使用的样本数
NUM_ROUNDS = 20
EVAL_INTERVAL = 1          # 每隔多少轮评估一次（1 = 每轮）

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

DATASET = "SHS27k"
SPLIT = "dataset/SHS27k_bfs.json"
OUT_DIR = Path("results")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print(f"[Experiment] Dataset={DATASET}, Split=bfs, Rounds={NUM_ROUNDS}")
    print(f"[Experiment] Train pool={TRAIN_POOL}, sampler={SAMPLER_SAMPLES}/round, "
          f"predictor={PREDICTOR_SAMPLES}/round")
    print(f"[Experiment] Device={config.device}")
    print()

    model = PPIModel.from_dataset(config, DATASET, "dataset")
    model.set_esm_tensor(f"dataset/{DATASET}_tensor.pt")
    split = model.load_split(SPLIT)

    train_idx = split["train_index"]
    val_idx = split["val_index"]
    test_idx = split["test_index"]

    # 固定一个评估子集（控制评估时长，验证集/测试集各取部分）
    rng = random.Random(42)
    val_subset = rng.sample(val_idx, min(400, len(val_idx)))
    test_subset = rng.sample(test_idx, min(400, len(test_idx)))

    history = []
    header = (f"{'Round':>5} | {'SamplerReward':>13} | {'TrainBCE':>8} | "
              f"{'ValAUC':>6} | {'ValF1@0.5':>8} | {'TestAUC':>7} | "
              f"{'TestF1@best':>10} | {'Thr':>5}")
    print(header)
    print("-" * len(header))

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()

        # 每轮随机抽样训练池
        pool = rng.sample(train_idx, TRAIN_POOL)

        # --- Sampler 训练步 ---
        sampler_info = model.train_sampler_step(pool[:SAMPLER_SAMPLES])

        # --- Predictor 训练步（训练集 loss = 该步 BCE） ---
        predictor_info = model.train_predictor_step(pool[:PREDICTOR_SAMPLES])

        # --- 评估：验证集上选最优阈值，应用到测试集 ---
        val_metrics = model.evaluate(val_subset)                      # F1@0.5
        best_th = model.tune_threshold(val_subset, average="macro")   # 验证集调阈值
        test_metrics = model.evaluate(test_subset, threshold=best_th) # 测试集用调后阈值
        test_f1_05 = model.evaluate(test_subset)["f1_macro"]          # 对比：测试集 F1@0.5

        elapsed = time.time() - t0
        rec = {
            "round": rnd,
            "sampler_reward": sampler_info["reward"],
            "policy_loss": sampler_info["policy_loss"],
            "value_loss": sampler_info["value_loss"],
            "train_bce": predictor_info["bce_loss"],
            "best_threshold": best_th,
            "val_auc_micro": val_metrics["auc_micro"],
            "val_f1_micro_05": val_metrics["f1_micro"],
            "val_f1_macro_05": val_metrics["f1_macro"],
            "test_auc_micro": test_metrics["auc_micro"],
            "test_f1_micro_best": test_metrics["f1_micro"],
            "test_f1_macro_best": test_metrics["f1_macro"],
            "test_f1_macro_05": test_f1_05,
            "elapsed_s": elapsed,
        }
        history.append(rec)

        print(
            f"{rnd:>5} | {rec['sampler_reward']:>13.4f} | {rec['train_bce']:>8.4f} | "
            f"{rec['val_auc_micro']:>6.4f} | {rec['val_f1_macro_05']:>8.4f} | "
            f"{rec['test_auc_micro']:>7.4f} | {rec['test_f1_macro_best']:>10.4f} | "
            f"{rec['best_threshold']:>5.2f}"
        )

    # 保存结果
    out_path = OUT_DIR / "shs27k_20rounds.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2, default=lambda x: float(x) if np.isscalar(x) else x)
    print(f"\n[Experiment] Results saved to {out_path}")

    # 汇总趋势
    print("\n[Experiment] Summary (first 3 rounds vs last 3 rounds):")
    for rnd in [1, 2, 3, NUM_ROUNDS - 2, NUM_ROUNDS - 1, NUM_ROUNDS]:
        rec = history[rnd - 1]
        print(
            f"  Round {rnd:>2}: TrainBCE={rec['train_bce']:.4f}, "
            f"SamplerReward={rec['sampler_reward']:.4f}, "
            f"ValAUC={rec['val_auc_micro']:.4f}, TestAUC={rec['test_auc_micro']:.4f}"
        )


if __name__ == "__main__":
    main()
