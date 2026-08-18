# SHS27k BFS：k_hops=1、REINFORCE gamma=1.0 实验报告

## 实验摘要

本实验使用当前 sampler 实现，在 SHS27k BFS 划分上训练 10 个 epoch。G0 仅由
target/proxy 构成，候选 action 限制在 G0 安全 adjacency 的 1-hop 区域内；
奖励使用相邻图损失增量，batch 内 advantage 做标准化。

最佳验证 Macro-AUC 出现在第 8 个 epoch，测试集只在该最佳 checkpoint 上评估
一次。

实验基于 `main` 基线提交 `a113945` 加当前工作区未提交的训练和 k-hop 改动，
结果不是单独提交 `a113945` 的复现结果。

## 配置与数据

```text
dataset: SHS27k
split: bfs
device: cuda (NVIDIA GeForce RTX 5090)
seed: 42
epochs: 10
batch_size / eval_batch_size: 32 / 64
hidden_dim: 256
GAT: 2 layers, 4 heads, dropout=0.1
sampler: max_steps=10, k_hops=1
sampler_lr / predictor_lr: 1e-4 / 1e-3
reinforce_baseline_coef: 0.1
reinforce_gamma: 1.0
F1 threshold: 0.5
```

BFS split 样本数为 train `4562`、validation `1524`、test `1538`。测试可见性
分组为 ES `1078`、NS `460`、BS `0`。

复现命令：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --device cuda \
  --epochs 10 \
  --reinforce-gamma 1.0 \
  --k-hops 1 \
  --seed 42 \
  --output /tmp/ppi_shs27k_bfs_khops1_gamma1_epoch10.json
```

原始 JSON 和 stdout 日志：

```text
/tmp/ppi_shs27k_bfs_khops1_gamma1_epoch10.json
/tmp/ppi_shs27k_bfs_khops1_gamma1_epoch10.log
```

## 训练曲线

| Epoch | Sampler loss | Predictor loss | Mean reward | Val Macro-AUC | Val Micro-AUC | Val Macro-F1 | Val Micro-F1 | 秒 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.042564 | 0.483136 | 0.003016 | 0.716877 | 0.739019 | 0.416171 | 0.482641 | 118.4 |
| 2 | 0.039159 | 0.367945 | 0.010710 | 0.715880 | 0.738815 | 0.392123 | 0.453255 | 113.1 |
| 3 | 0.000884 | 0.314960 | 0.015848 | 0.742277 | 0.753985 | 0.347884 | 0.415902 | 110.7 |
| 4 | -0.018021 | 0.274066 | 0.020705 | 0.723326 | 0.746755 | 0.345917 | 0.392420 | 111.0 |
| 5 | -0.023173 | 0.247236 | 0.025339 | 0.723978 | 0.745782 | 0.318315 | 0.385153 | 118.1 |
| 6 | -0.031025 | 0.219388 | 0.028232 | 0.737059 | 0.764523 | 0.511330 | 0.551145 | 114.2 |
| 7 | -0.018373 | 0.207288 | 0.031286 | 0.728476 | 0.755768 | 0.386967 | 0.443664 | 116.9 |
| **8** | **-0.027538** | **0.187248** | **0.034208** | **0.752648** | **0.778662** | **0.472729** | **0.517944** | 110.6 |
| 9 | -0.045180 | 0.172322 | 0.038300 | 0.713950 | 0.744173 | 0.372940 | 0.443412 | 110.0 |
| 10 | -0.047548 | 0.160382 | 0.037858 | 0.736075 | 0.759188 | 0.419468 | 0.494322 | 116.4 |

总耗时为 `1148.4s`（约 19.1 分钟）。Predictor loss 从 `0.483136` 降至
`0.160382`，mean reward 从 `0.003016` 上升至 `0.037858`。验证指标在 epoch
间波动明显，第 8 epoch 后 Macro-AUC 回落，说明 checkpoint 选择敏感。

## 最佳 checkpoint 测试结果

最佳 checkpoint：epoch 8，最佳 Val Macro-AUC `0.752648`。

| 测试分组 | 数量 | Macro-AUC | Micro-AUC | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| Overall | 1538 | **0.743714** | **0.769045** | **0.452583** | **0.497116** |
| ES（一端在训练节点中） | 1078 | 0.742673 | 0.765544 | 0.458524 | 0.501390 |
| NS（两端均不在训练节点中） | 460 | 0.746599 | 0.777559 | 0.444739 | 0.486579 |
| BS（两端均在训练节点中） | 0 | — | — | — | — |

本次 NS 的 AUC 略高于 ES，但两组样本量和训练动态不同，单次运行不足以说明
k-hop 限制对两类可见性样本存在稳定差异。BFS 没有 BS 样本，不能报告 BS 性能。

## 结论与限制

1. 在当前单次 seed=42 实验中，k_hops=1 的最佳 checkpoint 测试 Macro/Micro-AUC 为 `0.743714/0.769045`。
2. k-hop 限制将候选空间固定在 G0 的一跳安全邻域内，但本次实验没有与同一工作区、同一 seed 的其他 k 值做配对消融，不能据此判断 k=1 是否最优。
3. F1 使用固定阈值 `0.5`，对输出校准和 checkpoint 波动敏感；应结合多 seed AUC 统计评价最终效果。
4. 该结果为单次实验，原始输出位于 `/tmp`，正式复现时应归档 JSON、日志、命令、GPU/驱动和完整工作区 diff。
