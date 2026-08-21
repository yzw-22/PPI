# SHS27k BFS：learned sampler、gamma=1.0、epochs=10 实验记录（历史）

> **历史记录**：产生于旧版"初始邻居采样"机制（每个 seed 预采样一个邻居）
> 删除之前，当前 CLI 无法直接复现。完整对比请见
> [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)。

## 摘要

旧 G0 语义下的 learned sampler + GAT Predictor，SHS27k BFS，10 epochs，
gamma=1.0。最佳验证 Macro-AUC 在 epoch 7（0.743260），测试集只在该
checkpoint 上评估一次。含增量奖励与 batch advantage 标准化改动。

## 配置

```text
dataset/split: SHS27k / bfs          seed: 42
epochs: 10    batch/eval: 32/64      hidden_dim: 256
GAT: 2 layers, 4 heads, dropout=0.1  sampler: max_steps=10; 旧初始邻居机制
lr: 1e-4 / 1e-3                      gamma: 1.0   F1 阈值: 0.5
train/val/test: 4562 / 1524 / 1538   测试集可见性: ES 1078 / NS 460 / BS 0
```

奖励：`r_t = L(G_{t-1}) - L(G_t)`，第一步以 G0 为前项，无子图大小惩罚；
batch 内 advantage 标准化，value loss 回归未标准化 return-to-go。

## 最佳 checkpoint 测试结果（epoch 7）

| 分组 | 数量 | Macro-AUC | Micro-AUC | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| Overall | 1538 | 0.750387 | 0.772452 | 0.461202 | 0.532757 |
| ES | 1078 | 0.756029 | 0.783162 | 0.475677 | 0.545865 |
| NS | 460 | 0.736376 | 0.745707 | 0.426883 | 0.500826 |

总耗时 1233 s（~21 分钟）。ES 四项指标均高于 NS。

## 结论

1. 单 seed 单次实验，测试 MacAUC 0.7504 / MicAUC 0.7725，不作为稳定提升证据。
2. F1 固定阈值 0.5，受校准与 checkpoint 波动影响；AUC 更适合单次判断。
