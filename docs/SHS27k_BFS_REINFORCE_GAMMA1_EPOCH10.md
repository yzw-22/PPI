# SHS27k BFS：REINFORCE gamma=1.0 实验报告

## 实验摘要

本实验在 SHS27k 的 BFS 划分上训练 learned sampler 与 GAT Predictor，训练
10 个 epoch，`reinforce_gamma=1.0`。最佳验证 Macro-AUC 出现在第 7 个 epoch；
随后只在该最佳状态上评估一次测试集。

实验使用当前工作区的 `main` 基线提交 `a113945`，并包含工作区尚未提交的
增量奖励与 batch advantage 标准化改动。因此该结果对应的是当前工作区状态，
不是只由 `a113945` 提交复现的结果。该实验记录产生于初始邻居机制删除前，
不能直接视为当前 G0 语义下的结果；当前实现的 G0 只含 target/proxy，候选由
`k_hops` 区域限制。

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
sampler: max_steps=10; historical initial-neighbor sampling (one node per seed)
sampler_lr / predictor_lr: 1e-4 / 1e-3
reinforce_baseline_coef: 0.1
reinforce_gamma: 1.0
F1 threshold: 0.5
```

BFS split 样本数为 train `4562`、validation `1524`、test `1538`。测试可见性
分组为 ES `1078`、NS `460`、BS `0`；BS 为空是该 BFS 划分的结构属性。

采样器训练使用相邻图损失增量作为奖励：

\[
r_t=L(G_{t-1})-L(G_t),
\]

其中第一步的 `G(t-1)` 为 `G0`，不包含子图大小或新增节点数惩罚。每个 sampler
batch 的 detached advantage 统一标准化；value loss 仍使用未标准化的
return-to-go。

复现命令：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --device cuda \
  --epochs 10 \
  --reinforce-gamma 1.0 \
  --seed 42 \
  --output /tmp/ppi_shs27k_bfs_learned_gamma1_epoch10.json
```

上面的命令用于记录历史运行参数；旧版初始邻居机制已删除，不能在当前 CLI
上直接复现。当前语义下请使用 k-hop 实验报告中的命令。

原始 JSON 和 stdout 日志本次保存在：

```text
/tmp/ppi_shs27k_bfs_learned_gamma1_epoch10.json
/tmp/ppi_shs27k_bfs_learned_gamma1_epoch10.log
```

## 训练曲线

| Epoch | Sampler loss | Predictor loss | Mean reward | Val Macro-AUC | Val Micro-AUC | Val Macro-F1 | Val Micro-F1 | 秒 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.013704 | 0.492612 | 0.000052 | 0.680725 | 0.715957 | 0.331608 | 0.455491 | 126.9 |
| 2 | -0.006626 | 0.391593 | 0.000617 | 0.728311 | 0.744691 | 0.363420 | 0.472650 | 125.7 |
| 3 | -0.005151 | 0.336098 | 0.002138 | 0.730695 | 0.761733 | 0.470390 | 0.545180 | 121.6 |
| 4 | 0.002796 | 0.300365 | 0.002667 | 0.729126 | 0.757198 | 0.343308 | 0.393146 | 121.2 |
| 5 | -0.016251 | 0.267268 | 0.003456 | 0.738980 | 0.762654 | 0.386600 | 0.458497 | 123.1 |
| 6 | -0.006690 | 0.243534 | 0.003897 | 0.734680 | 0.767071 | 0.464530 | 0.502160 | 123.0 |
| **7** | **-0.006985** | **0.220147** | **0.004375** | **0.743260** | 0.753195 | 0.435214 | 0.496541 | 126.5 |
| 8 | -0.017000 | 0.198397 | 0.004294 | 0.738380 | 0.742644 | 0.391933 | 0.444128 | 117.6 |
| 9 | -0.012136 | 0.182861 | 0.004121 | 0.739887 | 0.748617 | 0.406507 | 0.481500 | 116.6 |
| 10 | -0.016762 | 0.176680 | 0.004706 | 0.728143 | 0.736707 | 0.431259 | 0.462885 | 121.3 |

总耗时为 `1233.1s`（约 20.6 分钟）。Predictor loss 从 `0.492612` 降至
`0.176680`，mean reward 总体由 `0.000052` 上升至 `0.004706`；验证
Macro-AUC 在第 7 epoch 达到峰值后回落，说明 checkpoint 选择对结果有影响。

## 最佳 checkpoint 测试结果

最佳 checkpoint：epoch 7，最佳 Val Macro-AUC `0.743260`。

| 测试分组 | 数量 | Macro-AUC | Micro-AUC | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| Overall | 1538 | **0.750387** | **0.772452** | **0.461202** | **0.532757** |
| ES（一端在训练节点中） | 1078 | 0.756029 | 0.783162 | 0.475677 | 0.545865 |
| NS（两端均不在训练节点中） | 460 | 0.736376 | 0.745707 | 0.426883 | 0.500826 |
| BS（两端均在训练节点中） | 0 | — | — | — | — |

ES 的四项指标均高于 NS，表明节点是否在训练图中出现仍是 BFS 泛化难度的
重要因素。由于 BS 样本为空，不能在 BFS 上报告 BS 性能或将其与 Random split
的 BS 结果比较。

## 结果解读与限制

1. 本次 run 在第 7 epoch 获得测试 Macro-AUC `0.750387`、Micro-AUC `0.772452`，但这是单 seed、单次实验，不能作为稳定性能提升的证据。
2. 验证 Macro-AUC 在 epoch 7 后下降，测试集只使用最佳验证 checkpoint，未参与训练期间模型选择。
3. F1 使用固定阈值 `0.5`，会受到输出校准和 checkpoint 波动影响；AUC 更适合用于当前单次运行的排序能力判断。
4. `/tmp` 不是长期归档位置；若需正式复现，应将 JSON、日志、命令、GPU/驱动和完整工作区 diff 一并归档。
