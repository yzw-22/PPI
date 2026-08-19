# SHS27k BFS：k_hops=1、gamma=1.0、epochs=10 实验记录（历史）

> 历史单次实验记录，完整对比请见 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)。

## 摘要

当前 sampler（G0 仅含 target/proxy，候选限于 G0 安全 1-hop 区域，增量损失
奖励 + batch advantage 标准化）在 SHS27k BFS 上训练 10 epochs、gamma=1.0。
最佳验证 Macro-AUC 在 epoch 8（0.752648），测试集只在该 checkpoint 上评估
一次。基于 `a113945` 加工作区未提交改动，非单独提交的复现。

## 配置

```text
dataset/split: SHS27k / bfs          seed: 42
epochs: 10    batch/eval: 32/64      hidden_dim: 256
GAT: 2 layers, 4 heads, dropout=0.1  sampler: max_steps=10, k_hops=1
lr: 1e-4 / 1e-3                      gamma: 1.0   F1 阈值: 0.5
train/val/test: 4562 / 1524 / 1538   可见性: ES 1078 / NS 460 / BS 0
```

复现命令：

```bash
python -m src.train_shs27k --dataset SHS27k --split bfs --device cuda \
  --epochs 10 --reinforce-gamma 1.0 --k-hops 1 --seed 42 \
  --output /tmp/ppi_shs27k_bfs_khops1_gamma1_epoch10.json
```

## 最佳 checkpoint 测试结果（epoch 8）

| 分组 | 数量 | Macro-AUC | Micro-AUC | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| Overall | 1538 | 0.743714 | 0.769045 | 0.452583 | 0.497116 |
| ES | 1078 | 0.742673 | 0.765544 | 0.458524 | 0.501390 |
| NS | 460 | 0.746599 | 0.777559 | 0.444739 | 0.486579 |

总耗时 1148 s（~19 分钟）。Predictor loss 0.483→0.160，mean reward
0.003→0.038。验证指标 epoch 间波动大，checkpoint 选择敏感。

## 结论

1. 单次 seed=42，k_hops=1 最佳测试 AUC 0.7437/0.7690。
2. 未与同 seed 其他 k 值配对消融，不能判定 k=1 最优。
3. 单次结果，正式复现需归档 JSON、日志、命令与完整 diff。
