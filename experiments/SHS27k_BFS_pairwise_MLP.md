# SHS27k BFS：Pairwise MLP Sampler 实验

## 配置

- 数据集/划分：SHS27k BFS
- `hidden_dim=512`
- `max_steps=10`
- `k_hops=3`
- `reinforce_gamma=0.95`
- GAT `num_layers=3`
- 训练轮数：10 Epoch
- F1 阈值：固定 `0.5`
- 评估图：每条轨迹的 `final_graph`

Sampler 的 action score 为：

\[
e_i = W_2\,\operatorname{LeakyReLU}
    (W_1[W_s s\Vert W_i x_i]+b_1)+b_2
\]

其中 state 和 candidate 使用不同的投影矩阵，pairwise MLP 的 Linear 权重使用 Xavier uniform 初始化。

## 最佳结果

每个 seed 按验证集 Macro-AUC 选择最佳 Epoch，再报告同一 Epoch 的测试结果。

| Seed | 最佳 Epoch | Val Macro-AUC | Val Micro-AUC | Val Macro-F1 | Val Micro-F1 | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9 | 0.7395 | 0.7563 | 0.5070 | 0.5576 | 0.7347 | 0.7607 | 0.5192 | 0.5629 |
| 123 | 10 | 0.7347 | 0.7506 | 0.4797 | 0.5365 | 0.7216 | 0.7452 | 0.4732 | 0.5184 |

测试集可见性分组：

| Seed | ES 数量 | ES Macro/Micro-AUC | ES Macro/Micro-F1 | NS 数量 | NS Macro/Micro-AUC | NS Macro/Micro-F1 | BS 数量 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1078 | 0.7471/0.7757 | 0.5455/0.5853 | 460 | 0.7045/0.7224 | 0.4542/0.5078 | 0 |
| 123 | 1078 | 0.7269/0.7528 | 0.4929/0.5357 | 460 | 0.7205/0.7259 | 0.4233/0.4733 | 0 |

## 训练曲线摘要

下表保留每个 Epoch 的 Sampler loss、Predictor loss、mean reward 和验证 Macro-AUC。

| Epoch | Sampler loss (42) | Predictor loss (42) | Reward (42) | Val AUC (42) | Sampler loss (123) | Predictor loss (123) | Reward (123) | Val AUC (123) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.0119 | 0.5236 | 0.0154 | 0.6413 | 0.0270 | 0.5019 | 0.0293 | 0.6494 |
| 2 | 0.0472 | 0.4346 | 0.0451 | 0.7211 | 0.0180 | 0.4243 | 0.0597 | 0.7088 |
| 3 | 0.0550 | 0.3800 | 0.0560 | 0.6774 | 0.0576 | 0.3738 | 0.0728 | 0.7082 |
| 4 | 0.0615 | 0.3399 | 0.0699 | 0.7363 | 0.0774 | 0.3322 | 0.0861 | 0.6899 |
| 5 | 0.0841 | 0.3048 | 0.0809 | 0.7091 | 0.0831 | 0.3012 | 0.0970 | 0.6934 |
| 6 | 0.0853 | 0.2785 | 0.0871 | 0.7299 | 0.1059 | 0.2811 | 0.1071 | 0.6798 |
| 7 | 0.1153 | 0.2581 | 0.1007 | 0.6861 | 0.1394 | 0.2539 | 0.1237 | 0.7171 |
| 8 | 0.1289 | 0.2419 | 0.1053 | 0.7139 | 0.1511 | 0.2429 | 0.1295 | 0.7056 |
| 9 | 0.1508 | 0.2345 | 0.1146 | 0.7395 | 0.1544 | 0.2284 | 0.1353 | 0.7159 |
| 10 | 0.1409 | 0.2149 | 0.1249 | 0.6898 | 0.2042 | 0.2129 | 0.1532 | 0.7347 |

## 结论与限制

- 两个 seed 的最佳验证 Macro-AUC 为 `0.7395` 和 `0.7347`，测试 Macro-AUC 为 `0.7347` 和 `0.7216`，结果存在明显随机波动。
- Predictor loss 总体持续下降，但验证性能非单调，说明不能以训练损失选择 checkpoint；应按验证 Macro-AUC 保存最佳 checkpoint。
- 每次实验耗时约 25 分钟。pairwise MLP 对每一步的所有候选逐一评分，是当前主要计算开销之一。
- BFS 测试集 BS 分组为空；ES/NS 结果不能直接与其他划分的整体指标等价比较。
- 当前只有两个 seed，尚不足以给出稳定的均值、标准差或显著性结论。

复现实验命令：

```bash
python -m src.train_shs27k \
  --split bfs \
  --epochs 10 \
  --hidden-dim 512 \
  --max-steps 10 \
  --k-hops 3 \
  --reinforce-gamma 0.95 \
  --gnn-layers 3 \
  --seed 42
```

将 `--seed` 改为 `123` 可复现第二次实验。完整逐 epoch JSON 结果曾保存在：

- `/tmp/shs27k_bfs_steps10_gamma095_gat3_pairwise_mlp.json`
- `/tmp/shs27k_bfs_steps10_gamma095_gat3_pairwise_mlp_seed123.json`
