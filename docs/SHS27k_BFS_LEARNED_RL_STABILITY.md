# SHS27k BFS Learned Sampler：RL 稳定性实验总结

## 实验目的

本实验用于评估加入 batch advantage 标准化后的 learned sampler 在 REINFORCE 训练中的数值稳定性、策略行为和跨运行波动。除随机 seed 外，所有配置保持一致。

## 配置

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 10 \
  --seed <seed> \
  --output <output.json>
```

默认配置为 batch size `32`、eval batch size `64`、hidden dim `256`、GAT `2` 层 / `4` heads、dropout `0.1`、sampler/predictor 学习率 `1e-4/1e-3`、`reinforce_gamma=1.0`、`complexity_penalty=0`。实验使用 seed `42` 两次重复，以及 seed `43`、`44`、`45` 各一次，共 5 次运行。

本次 sampler policy loss 使用同一 batch 全部动作 step 的 detached advantage 做总体标准化；value loss 仍回归原始 return-to-go。额外记录了 raw advantage、策略熵、STOP advantage、sampler 梯度范数和对应计数。诊断指标不参与优化，也没有启用梯度裁剪或 entropy bonus。

## 运行结果

| 运行 | 最佳 epoch | 最佳 Val Macro-AUC | 最佳 checkpoint Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 总耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42（首次） | 10 | 0.7479 | 0.7283 | 0.7653 | 0.4984 | 0.5599 | 1463.67 |
| seed 42（重复） | 10 | 0.7428 | 0.7404 | 0.7696 | 0.4837 | 0.5355 | 1466.65 |
| seed 43 | 8 | 0.7360 | 0.7147 | 0.7420 | 0.4061 | 0.4503 | 1470.65 |
| seed 44 | 2 | 0.7447 | 0.7277 | 0.7501 | 0.4198 | 0.4868 | 1508.12 |
| seed 45 | 10 | 0.7568 | 0.7495 | 0.7583 | 0.4682 | 0.5139 | 1425.36 |
| **均值** | — | **0.7456** | **0.7321** | **0.7571** | **0.4552** | **0.5093** | **1466.89** |
| **样本标准差** | — | **0.0076** | **0.0133** | **0.0113** | **0.0385** | **0.0429** | **29.97** |

最佳验证 Macro-AUC 范围为 `0.7360–0.7568`。同 seed 42 两次的最佳验证 Macro-AUC 相差 `0.0050`；对应测试 Macro-AUC 相差 `0.0121`，说明 checkpoint 泛化仍存在明显运行间波动。

## RL 训练稳定性

下表比较每次运行第 1 个和第 10 个 epoch 的关键指标；范围列覆盖全部 5 次运行的全部 epoch。

| 指标 | Epoch 1 范围 | Epoch 10 范围 | 全部运行/epoch 范围 | 观察 |
|---|---:|---:|---:|---|
| mean reward | 0.0016–0.0029 | 0.0103–0.0110 | 0.0016–0.0110 | 平滑上升，没有符号反复或爆炸 |
| raw advantage std | 0.0506–0.0599 | 0.1010–0.1163 | 0.0506–0.1163 | 方差逐步增大，但 batch 标准化后 policy 更新仍有限 |
| value loss | 0.0026–0.0036 | 0.0102–0.0135 | 0.0026–0.0135 | 上升后趋于有限范围，未发散 |
| policy entropy | 3.28–3.90 | 0.53–0.76 | 0.53–3.90 | 明显下降，策略逐渐变尖锐 |
| sampler grad norm | 0.2165–0.2409 | 0.1717–0.2091 | 0.1679–0.2452 | 未出现梯度尖峰或持续增长 |
| mean steps | 9.92–9.97 | 9.90–9.98 | 9.90–9.99 | 始终接近 `max_steps=10` |
| STOP rate | 0.0040–0.0134 | 0.0004–0.0197 | 0–0.0197 | 多数运行接近零，seed 44 后期略高 |

所有 50 个 epoch 记录中的 RL 诊断值均为有限数，没有 NaN/Inf。raw advantage 的 batch 均值始终接近 0，说明 value baseline 的中心位置没有明显漂移；梯度范数整体下降或波动，没有数值爆炸。

## 结论

1. **数值训练稳定。** batch advantage 标准化后，policy loss、value loss、raw advantage 和 sampler 梯度均保持有限；未观察到梯度爆炸、value loss 发散或 NaN。
2. **存在明显的策略熵塌缩/行为饱和。** 熵从约 `3.3–3.9` 降至 `0.53–0.76`，平均轨迹长度始终约为 10 步，说明策略倾向于执行满步扩展，而不是学习稳定的提前 STOP 决策。
3. **跨运行泛化仍有波动。** 最佳验证 Macro-AUC 的样本标准差为 `0.0076`，测试 Macro-AUC 的样本标准差为 `0.0133`；seed 44 在 epoch 2 达到最佳，而其他运行多在 epoch 8–10 达到最佳，checkpoint 选择较敏感。
4. **当前改动降低了 REINFORCE 的数值风险，但没有解决策略行为问题。** 后续应优先做单变量消融：小幅 entropy bonus、STOP reward/penalty 设计或策略熵监控下的早停；同时保留 batch advantage 标准化作为基础配置。

## `reinforce_gamma=0.95` 对照实验

在相同 seed `42`、数据划分和其他超参下，将 `reinforce_gamma` 从 `1.0` 改为 `0.95`，运行命令为：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 10 \
  --seed 42 \
  --reinforce-gamma 0.95 \
  --output /tmp/ppi_shs27k_bfs_learned_gamma095_seed42.json
```

| gamma | 最佳 epoch | 最佳 Val Macro-AUC | 最佳 Val Micro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 总耗时 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 10 | 0.7479 | 0.7783 | 0.7283 | 0.7653 | 0.4984 | 0.5599 | 1463.67 |
| 0.95 | 10 | 0.7507 | 0.7750 | 0.7266 | 0.7639 | 0.5046 | 0.5708 | 1487.83 |
| **0.95 − 1.00** | — | **+0.0028** | **−0.0033** | **−0.0017** | **−0.0014** | **+0.0062** | **+0.0109** | **+24.15** |

gamma=0.95 的验证 Macro-AUC 略高，但测试 Macro/Micro-AUC 分别低 `0.0017/0.0014`；测试 Macro/Micro-F1 分别高 `0.0062/0.0109`。因此单次 seed=42 不能证明 gamma=0.95 带来总体性能提升，差异更像 checkpoint 泛化和阈值 F1 波动。

RL 诊断方面，两种 gamma 都保持有限梯度和 value loss。gamma=0.95 的 sampler 梯度范围为 `0.1810–0.2220`，gamma=1.0 为 `0.1875–0.2452`；最终策略熵分别为 `0.6946` 和 `0.7318`，平均步数分别为 `9.987` 和 `9.980`。两者都出现低熵、近满步、低 STOP 的策略饱和，gamma=0.95 未解决该行为问题。若要判断 gamma 的真实收益，应使用相同的多 seed 配对实验，而不能依据这一次运行定论。

gamma=0.95 的原始结果和日志为 `/tmp/ppi_shs27k_bfs_learned_gamma095_seed42.json` 与 `/tmp/ppi_shs27k_bfs_learned_gamma095_seed42.log`。

## complexity penalty `5e-4` 对照实验

在 gamma=0.95、seed=42 的基础上加入 `--complexity-penalty 5e-4`：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 10 \
  --seed 42 \
  --reinforce-gamma 0.95 \
  --complexity-penalty 5e-4 \
  --output /tmp/ppi_shs27k_bfs_learned_gamma095_penalty0005_seed42.json
```

| 配置 | 最佳 epoch | 最佳 Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | mean steps（末轮） | STOP rate（末轮） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma=0.95, penalty=0 | 10 | 0.7507 | 0.7266 | 0.7639 | 0.5046 | 0.5708 | 9.987 | 0.0000 |
| gamma=0.95, penalty=5e-4 | 6 | 0.7576 | 0.7642 | 0.7774 | 0.4802 | 0.5146 | 9.981 | 0.0007 |
| **penalty − no penalty** | — | **+0.0070** | **+0.0376** | **+0.0135** | **−0.0244** | **−0.0562** | **−0.006** | **+0.0007** |

在这次 seed=42 运行中，复杂度惩罚令最佳验证 epoch 提前到 6，并提升了按最佳验证 checkpoint 评估的 AUC；但两个 F1 指标下降，说明固定 `0.5` 阈值下的分类校准不同。轨迹长度只减少约 `0.005–0.01` 步，STOP rate 仍低于 `1%`，因此 `5e-4` 尚未实质解决满步策略饱和。该结果应视为单次消融信号，需多 seed 配对实验确认。

惩罚实验原始结果和日志为 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty0005_seed42.json` 与 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty0005_seed42.log`。

## complexity penalty `5e-3`（5 epoch）

继续沿用 gamma=0.95、seed=42，仅将惩罚提高到 `5e-3` 并运行 5 个 epoch：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 5 \
  --seed 42 \
  --reinforce-gamma 0.95 \
  --complexity-penalty 5e-3 \
  --output /tmp/ppi_shs27k_bfs_learned_gamma095_penalty005_seed42_epoch5.json
```

| 配置 | 最佳 epoch | 最佳 Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 末轮 mean steps | 末轮 STOP rate | 末轮策略熵 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma=0.95, penalty=5e-4 | 6 | 0.7576 | 0.7642 | 0.7774 | 0.4802 | 0.5146 | 9.981 | 0.0007 | 0.6067 |
| gamma=0.95, penalty=5e-3 | 2 | 0.7267 | 0.7347 | 0.7599 | 0.3466 | 0.4450 | 0.999 | 0.9989 | 0.0013 |
| **5e-3 − 5e-4** | — | **−0.0309** | **−0.0295** | **−0.0175** | **−0.1336** | **−0.0696** | **−8.982** | **+0.9982** | **−0.6054** |

`5e-3` 在第 1 个 epoch 就将 STOP rate 推高到 `0.7994`，第 2–5 epoch 稳定在约 `0.9989`，平均轨迹退化为约 1 步，策略熵接近 0。梯度范数从第 1 epoch 的 `0.568` 很快降至第 5 epoch 的 `0.0009`，没有数值爆炸，但这属于策略塌缩而非有效的 RL 稳定。相较 `5e-4`，AUC 和 F1 均明显下降，不建议继续使用该惩罚量级。

5 epoch 实验的原始结果和日志为 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty005_seed42_epoch5.json` 与 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty005_seed42_epoch5.log`。

## complexity penalty `1e-3`（5 epoch）

在 gamma=0.95、seed=42 下使用中等惩罚并运行 5 个 epoch：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 5 \
  --seed 42 \
  --reinforce-gamma 0.95 \
  --complexity-penalty 1e-3 \
  --output /tmp/ppi_shs27k_bfs_learned_gamma095_penalty001_seed42_epoch5.json
```

| 配置 | 最佳 epoch | 最佳 Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 末轮 mean steps | 末轮 STOP rate | 末轮策略熵 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma=0.95, penalty=5e-4 | 6 | 0.7576 | 0.7642 | 0.7774 | 0.4802 | 0.5146 | 9.981 | 0.0007 | 0.6067 |
| gamma=0.95, penalty=1e-3 | 5 | 0.7403 | 0.7218 | 0.7594 | 0.4941 | 0.5386 | 9.929 | 0.0110 | 1.0229 |
| **1e-3 − 5e-4** | — | **−0.0173** | **−0.0424** | **−0.0180** | **+0.0139** | **+0.0239** | **−0.053** | **+0.0103** | **+0.4161** |

`1e-3` 没有造成 `5e-3` 那样的立即 STOP 塌缩：5 个 epoch 内平均步数仍约 `9.93`，末轮 STOP rate 为 `1.10%`，策略熵为 `1.02`。但相较 `5e-4`，本次运行的最佳验证和测试 AUC 明显下降，F1 略有改善，体现出 AUC 与固定 `0.5` 阈值 F1 的权衡。结合 `5e-3` 的塌缩结果，当前单 seed 证据更支持较小惩罚；`1e-3` 的收益仍需多 seed 配对验证。

1e-3 实验原始结果和日志为 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty001_seed42_epoch5.json` 与 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty001_seed42_epoch5.log`。

## complexity penalty `2e-3`（5 epoch）

在 gamma=0.95、seed=42 下将惩罚提高到 `2e-3` 并运行 5 个 epoch：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 5 \
  --seed 42 \
  --reinforce-gamma 0.95 \
  --complexity-penalty 2e-3 \
  --output /tmp/ppi_shs27k_bfs_learned_gamma095_penalty002_seed42_epoch5.json
```

| 配置 | 最佳 epoch | 最佳 Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 末轮 mean steps | 末轮 STOP rate | 末轮策略熵 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma=0.95, penalty=1e-3 | 5 | 0.7403 | 0.7218 | 0.7594 | 0.4941 | 0.5386 | 9.929 | 0.0110 | 1.0229 |
| gamma=0.95, penalty=2e-3 | 3 | 0.7378 | 0.7329 | 0.7516 | 0.4980 | 0.5555 | 0.999 | 0.9989 | 0.0027 |
| **2e-3 − 1e-3** | — | **−0.0025** | **+0.0111** | **−0.0078** | **+0.0039** | **+0.0169** | **−8.929** | **+0.9879** | **−1.0202** |

`2e-3` 在第 1 个 epoch 就把 STOP rate 推高到 `0.7448`，第 2 个 epoch 起达到约 `0.9989`，平均轨迹退化为 1 步，策略熵接近 0。第 1 个 epoch 的 sampler 梯度范数达到 `1.21`，随后快速下降；没有 NaN/Inf，但这是明显的策略塌缩。相较 `1e-3`，AUC 变化不一致、F1 仅小幅改善，不能抵消近乎全 STOP 的行为退化；不建议使用该惩罚量级。

2e-3 实验原始结果和日志为 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty002_seed42_epoch5.json` 与 `/tmp/ppi_shs27k_bfs_learned_gamma095_penalty002_seed42_epoch5.log`。

## 原始结果与日志

结果文件位于 `/tmp`，包括：

- `/tmp/ppi_shs27k_bfs_learned.json`（seed 42 首次）
- `/tmp/ppi_shs27k_bfs_learned_seed42_repeat2.json`
- `/tmp/ppi_shs27k_bfs_learned_seed43.json`
- `/tmp/ppi_shs27k_bfs_learned_seed44.json`
- `/tmp/ppi_shs27k_bfs_learned_seed45.json`

对应的完整 stdout 日志为同名 `.log` 文件。由于 `/tmp` 不是长期归档目录，正式复现实验应将这些 JSON、日志、commit、环境和命令复制到版本化结果目录。
