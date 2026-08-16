# SHS27k BFS 实验汇总

## 1. 实验范围与统一设置

本汇总合并当前分支（`ablation`）和 main 分支上的 SHS27k BFS 实验。除特别说明外，实验使用：

- 10 个 epoch、CUDA、batch size `32`、eval batch size `64`、seed `42`；
- hidden size `256`、2 层 GAT、4 heads、dropout `0.1`；
- `fixed_num=1`、`max_steps=10`、Sampler/Predictor 学习率 `1e-4/1e-3`；
- 测试集 1538 个 PPI：ES 1078、NS 460、BS 0；
- 按验证集 Macro-AUC 选择 checkpoint，再评估测试集；F1 使用阈值 `0.5`。

ES/NS/BS 表示测试 PPI 两端相对于训练节点集合的可见性：ES 为一端可见、NS 为两端不可见、BS 为两端均可见。本 BFS 测试集没有 BS 样本。

当前分支的 learned sampler 使用相邻图增量奖励：

\[
r_t=L_{t-1}-L_t-\lambda\Delta n_t,
\]

其中 `Δn_t` 是本步新增的非 baseline 节点数，DONE/STOP 奖励为 `0`。main 分支使用旧的相对 `G_0` 奖励且没有显式 STOP，因此 main 与当前分支不是单变量比较。

典型实验命令为：

```bash
python -m src.train_shs27k \
  --dataset SHS27k --split bfs --sampler-mode learned \
  --device cuda --epochs 10 --seed <SEED> \
  --reinforce-gamma <GAMMA> \
  --complexity-penalty <LAMBDA> \
  --output /tmp/<RESULT>.json
```

main 分支实验使用提交 `a113945`，额外指定了现有数据目录和 cache 目录。

## 2. 主结果对比

| 实现/配置 | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| main 旧实现，γ=1.0 | 10 | 0.7413 | 0.7279 | 0.7546 | 0.4780 | 0.5029 | 1188.9 |
| 当前分支，λ=0、γ=1.0 | 4 | **0.7528** | 0.7479 | **0.7877** | **0.5465** | **0.6070** | 2024.6 |
| 当前分支，λ=0、γ=0.9 | 8 | 0.7501 | 0.7417 | 0.7777 | 0.5446 | 0.5895 | 1885.2 |
| 当前分支，λ=0、γ=0.95 | 9 | **0.7580** | 0.7467 | 0.7664 | 0.5000 | 0.5458 | 1408.2 |

当前分支的 gamma=1.0 相比 main 测试 Macro-AUC、Micro-AUC、Macro-F1、Micro-F1 分别提高约 `0.0200`、`0.0331`、`0.0685`、`0.1041`，但运行时间增加约 70%。由于 reward、STOP、proxy/cache、邻接和 sampler 逻辑均有变化，这只是版本级结果对比，不能归因于某一项改动。

### 固定 sampler 基线

以下均为当前分支、BFS、seed=42、γ=1.0、λ=0 的参考结果：

| sampler | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 4 | **0.7528** | **0.7479** | **0.7877** | **0.5465** | **0.6070** | 2024.6 |
| target_only | 9 | 0.7389 | 0.7422 | 0.7635 | 0.5181 | 0.5331 | 35.2 |
| target_proxy | 10 | 0.7366 | 0.7426 | 0.7675 | 0.5246 | 0.5595 | 60.7 |
| random_1hop10 | 3 | 0.7503 | 0.7456 | 0.7578 | 0.3378 | 0.4290 | 66.6 |
| random_iterative10 | 9 | 0.7508 | 0.7370 | 0.7529 | 0.4862 | 0.5003 | 237.1 |

learned 的主要优势是 Micro-AUC/F1 和总体排序，但 Macro-AUC 相对固定基线的领先幅度很小；其代价是约 8～57 倍的运行时间。random_iterative10 与 learned 的图规模接近但效果较差，说明节点选择质量比单纯增加节点更重要。

## 3. gamma 消融

在 λ=0、seed=42 下，gamma=0.9 和 0.95 都没有稳定优于 gamma=1.0：

- gamma=0.9：测试 Micro-AUC、Macro-F1、Micro-F1 分别比 gamma=1.0 低约 `0.0100`、`0.0019`、`0.0175`；NS 组下降更明显。
- gamma=0.95：验证 Macro-AUC 最高（`0.7580`），但测试 Micro-AUC/F1 明显低于 gamma=1.0。
- 三种 gamma 的平均步数都约为 `9.97～9.98`，STOP rate 均低于 `1.1%`，仅调整折扣因子不能解决提前 DONE。

当前默认 gamma 仍建议使用 `1.0`，除非多 seed 实验证明折扣回报具有稳定收益。

## 4. complexity penalty 消融

### 4.1 总体指标

| λ | seed | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 最佳步数 / STOP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 4 | 0.7528 | 0.7479 | 0.7877 | 0.5465 | 0.6070 | 2024.6 | 9.97 / 0.22% |
| 0.001 | 42 | 10 | 0.7449 | 0.7272 | 0.7516 | 0.4573 | 0.4794 | 1301.2 | 9.07 / 21.2% |
| 0.001 | 111 | 3 | **0.7537** | 0.7416 | 0.7365 | 0.4010 | 0.4380 | 1097.1 | 9.18 / 18.6% |
| 0.001 | 123 | 9 | 0.7385 | 0.7430 | 0.7707 | 0.5177 | 0.5680 | 271.2 | 1.00 / 99.89% |
| 0.0015 | 123 | 6 | 0.7393 | 0.7415 | 0.7732 | 0.4730 | 0.5190 | 244.0 | 1.00 / 99.89% |
| 0.002 | 111 | 8 | 0.7414 | 0.7251 | 0.7603 | 0.4780 | 0.5342 | 247.4 | 1.00 / 99.89% |
| 0.005 | 42 | 4 | 0.7531 | **0.7523** | 0.7704 | 0.4788 | 0.5131 | 284.7 | 1.16 / 99.89% |

复杂度惩罚在当前 BCE/reward 尺度下影响非常陡峭：λ=0.005 几乎立即停止；λ=0.001 在不同 seed 间既可能保持约 9 步，也可能第二个 epoch 起退化为 1 步。λ=0.0015 和 0.002 在已有 seed 上均属于强早停区间。

### 4.2 关键 ES/NS 结果

| 配置 | ES Macro/Micro-AUC | NS Macro/Micro-AUC | ES Macro/Micro-F1 | NS Macro/Micro-F1 |
|---|---:|---:|---:|---:|
| λ=0、γ=1.0、seed=42 | 0.7472 / 0.7846 | 0.7505 / 0.7952 | 0.5472 / 0.6033 | 0.5382 / 0.6161 |
| λ=0.005、seed=42 | 0.7588 / 0.7813 | 0.7389 / 0.7428 | 0.4971 / 0.5338 | 0.4285 / 0.4616 |
| λ=0.001、seed=111 | 0.7433 / 0.7481 | 0.7394 / 0.7080 | 0.4252 / 0.4631 | 0.3408 / 0.3772 |
| λ=0.001、seed=123 | 0.7605 / 0.7884 | 0.7034 / 0.7259 | 0.5467 / 0.5897 | 0.4435 / 0.5149 |
| λ=0.0015、seed=123 | 0.7501 / 0.7805 | 0.7239 / 0.7550 | 0.4919 / 0.5350 | 0.4244 / 0.4788 |
| λ=0.002、seed=111 | 0.7455 / 0.7804 | 0.6734 / 0.7082 | 0.5061 / 0.5598 | 0.4035 / 0.4691 |

NS 结果对 seed 和惩罚系数较敏感；BS 为空，无法评估双端均见于训练图的泛化能力。

## 5. 已确认的问题

1. **训练目标与泛化脱钩**：Predictor loss 和 mean reward 通常下降/上升，但验证 Macro-AUC 在 epoch 间明显波动，最佳 epoch 可能很早或很晚。
2. **REINFORCE 方差较大**：同一 λ=0.001 在 seed=42、111、123 的最佳 epoch、STOP rate 和测试 F1 差异显著，不能依据单个 seed 选择 λ。
3. **STOP 信号不连续**：无惩罚时几乎永不 DONE；λ≥0.0015 的已有实验通常在第二个 epoch 后近乎全停；中间步数区间尚未稳定出现。
4. **learned 计算成本高**：轨迹中的多次 Predictor 前向、动态图构造和 Python frontier 操作主导耗时。
5. **NS/BS 评估不足**：NS 泛化不稳定，BS 在 BFS 测试集为空。
6. **版本对比存在混杂因素**：main 与当前分支同时改变了 reward、STOP、缓存、邻接和 sampler 逻辑，不能用作单项因果结论。

## 6. 优化方向

### 实验与奖励

- 对 λ=`{0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005}` 使用至少 3～5 个相同 seed，报告均值、标准差和 Pareto 曲线（性能、步数、耗时）。
- 记录每个 action 的扩展 reward、STOP advantage、value loss 和动作熵，区分“有效早停”和“策略塌缩为全停”。
- 对 reward/advantage 做标准化或裁剪；考虑按 baseline loss 归一化 penalty，避免 λ 直接与 BCE 绝对尺度绑定。
- 使用 entropy regularization、温度或显式目标步数/步数成本，稳定 3～6 步的中等轨迹。

### 计算优化

- 缓存 split/target 级别的 `G0`、归一化 embedding、candidate projection 和 adjacency。
- 增量维护 selected embedding/state，减少每步重复计算。
- 批量计算 step graph 的 Predictor 前向，减少重复 GAT 调用。
- 使用 `torch.inference_mode()`、AMP/BF16，并减少 Python set/list 与动态 tensor 构造。

### 模型与评估

- 对 class imbalance 测试 weighted BCE 或 focal loss；在验证集校准 F1 threshold，而不是固定使用 0.5。
- 增加 DFS 和其他 split，单独报告 ES/NS；若需要 BS 结论，应使用包含 BS 样本的 split。
- 保留验证 Macro-AUC checkpoint 选择，同时报告 Micro-F1 等业务指标，避免用最终 epoch 或单一指标替代验证选择。

## 7. 结论与推荐默认值

- 精度优先：暂保留 `complexity_penalty=0`、`reinforce_gamma=1.0`，因为当前单 seed 下 Micro-AUC/F1 最稳定且最高。
- 成本优先：λ=0.005 可将运行时间降至约 285 秒，但会让 sampler 几乎完全 DONE，不应视为通用默认值。
- λ=0.001～0.002 目前没有稳定的中间行为，需先完成多 seed 和 STOP advantage 诊断。
- learned sampler 相对固定基线的精度收益有限，扩大实验规模前应先优化计算路径并进行严格配对消融。

原始 JSON 保留在 `/tmp/ppi_shs27k_*.json`；本汇总保留历史数值，即使部分输出文件后来被同一路径实验覆盖，也不会影响表中记录。

## 8. DFS 实验（本次 `run.sh` 运行）

本次将 `run.sh` 的 split 从 BFS 改为 DFS 后，按原配置顺序运行五种 sampler：10 epochs、CUDA、seed=42、`reinforce_gamma=1.0`、`complexity_penalty=0`。结果文件为 `/tmp/ppi_shs27k_dfs_*.json`。DFS 测试集共 1529 个 PPI，其中 ES=1215、NS=314、BS=0。

### 8.1 总体结果

| sampler | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 平均步数 | STOP rate | 最终节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 10 | 0.8153 | 0.8095 | 0.8250 | 0.5782 | 0.6189 | 1389.8 | 9.96 | 0.72% | 14.02 |
| target_only | 10 | 0.8265 | 0.8212 | **0.8430** | 0.5966 | **0.6608** | 31.8 | 0.00 | 0% | 2.00 |
| target_proxy | 10 | 0.8255 | **0.8211** | 0.8379 | **0.6026** | 0.6534 | 38.4 | 0.00 | 0% | 2.09 |
| random_1hop10 | 9 | 0.8139 | 0.8139 | 0.8230 | 0.5746 | 0.6176 | 42.3 | 0.00 | 0% | 11.68 |
| random_iterative10 | 10 | 0.8169 | 0.8091 | 0.8238 | 0.5795 | 0.6208 | 143.4 | 9.99 | 0% | 14.06 |

### 8.2 ES/NS/BS 分组结果

下表为测试集分组的 Macro/Micro-AUC 与 Macro/Micro-F1；BS 由于没有样本，所有指标均未定义。

| sampler | ES AUC (Macro/Micro) | NS AUC (Macro/Micro) | ES F1 (Macro/Micro) | NS F1 (Macro/Micro) |
|---|---:|---:|---:|---:|
| learned | 0.8244 / 0.8394 | 0.7411 / 0.7565 | 0.6049 / 0.6409 | 0.4589 / 0.5163 |
| target_only | 0.8378 / 0.8588 | 0.7432 / 0.7657 | 0.6214 / 0.6866 | 0.4809 / 0.5467 |
| target_proxy | 0.8359 / 0.8515 | **0.7522 / 0.7731** | **0.6228 / 0.6728** | **0.5134 / 0.5663** |
| random_1hop10 | 0.8270 / 0.8344 | 0.7526 / 0.7700 | 0.5930 / 0.6381 | 0.4876 / 0.5255 |
| random_iterative10 | 0.8223 / 0.8340 | 0.7430 / 0.7761 | 0.5966 / 0.6413 | 0.4995 / 0.5286 |

### 8.3 与 BFS 的对比和结论

- DFS 的测试指标整体高于此前 BFS：learned 的 Test Macro/Micro-AUC 从 `0.7479/0.7877` 提升到 `0.8095/0.8250`，Macro/Micro-F1 从 `0.5465/0.6070` 提升到 `0.5782/0.6189`。这反映 split 的训练/测试节点可见性不同，不应解释为 sampler 改动带来的因果提升。
- DFS 中固定 sampler 明显优于 learned：`target_only` 的 Macro-AUC 最高（0.8212），`target_proxy` 的 Macro-F1 最高（0.6026），同时耗时仅约 32–38 秒；learned 耗时约 1389.8 秒，约为固定 sampler 的 33–44 倍。
- learned 与 random_iterative10 的最终图规模接近（约 14 个节点、约 10 步），但 learned 的 Macro-AUC/F1 略高，说明学习到的节点选择仍有收益，但收益不足以抵消当前计算成本。
- DFS 的 NS 指标仍低于 ES，尤其 learned 的 NS Macro-F1 仅 0.4589；BS 为空，无法评价双端节点均见于训练集的泛化。
- 本次 DFS 结果支持先采用固定 sampler 作为低成本基线，并优先优化 learned sampler 的 predictor 前向、动态图构造和轨迹采样开销；learned 的早停率仍低于 1%，说明在 `lambda=0` 下策略基本执行满步。

## 9. Random split 实验（本次 `run.sh` 运行）

本次将 `run.sh` 的 split 从 DFS 改为 `random`，其余配置保持不变：10 epochs、CUDA、seed=42、`reinforce_gamma=1.0`、`complexity_penalty=0`。结果文件为 `/tmp/ppi_shs27k_random_*.json`。随机划分测试集共 1525 个 PPI，其中 BS=1377、ES=138、NS=10；NS 样本过少，所有配置的 NS Macro-AUC 均不可定义。

### 9.1 总体结果

| sampler | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 平均步数 | STOP rate | 最终节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 10 | 0.9054 | 0.9092 | 0.9173 | 0.7251 | 0.7608 | 1385.9 | 9.98 | 0.04% | 14.06 |
| target_only | 10 | 0.9391 | 0.9413 | **0.9503** | **0.7875** | **0.8255** | 31.3 | 0.00 | 0% | 2.00 |
| target_proxy | 10 | 0.9371 | **0.9417** | 0.9493 | 0.7802 | 0.8240 | 36.6 | 0.00 | 0% | 2.09 |
| random_1hop10 | 10 | 0.9103 | 0.9198 | 0.9314 | 0.7266 | 0.7830 | 43.3 | 0.00 | 0% | 11.70 |
| random_iterative10 | 9 | 0.9097 | 0.9207 | 0.9260 | 0.7190 | 0.7649 | 149.7 | 9.98 | 0% | 14.06 |

### 9.2 BS/ES/NS 分组结果

| sampler | BS AUC (Macro/Micro) | ES AUC (Macro/Micro) | NS AUC (Macro/Micro) | BS F1 (Macro/Micro) | ES F1 (Macro/Micro) | NS F1 (Macro/Micro) |
|---|---:|---:|---:|---:|---:|---:|
| learned | 0.9150 / 0.9222 | 0.8495 / 0.8707 | — / 0.7081 | 0.7412 / 0.7744 | 0.4414 / 0.6052 | 0.1071 / 0.3750 |
| target_only | 0.9485 / 0.9567 | 0.8473 / 0.8758 | — / 0.7037 | **0.8052 / 0.8437** | 0.5614 / 0.6255 | 0.1524 / 0.3871 |
| target_proxy | **0.9494 / 0.9561** | 0.8537 / 0.8700 | — / 0.5649 | 0.7958 / 0.8397 | **0.5834 / 0.6430** | 0.0816 / 0.2963 |
| random_1hop10 | 0.9263 / 0.9371 | 0.8462 / 0.8726 | — / 0.6881 | 0.7419 / 0.7967 | 0.4670 / 0.6203 | 0.1973 / 0.4444 |
| random_iterative10 | 0.9266 / 0.9310 | **0.8657 / 0.8762** | — / 0.6548 | 0.7302 / 0.7785 | 0.5532 / 0.6032 | 0.0879 / 0.2857 |

### 9.3 结论与跨 split 对比

- Random split 的总体指标明显高于 BFS/DFS，主要原因是 BS 样本占测试集约 90%，而 BFS 没有 BS、DFS 也没有 BS；因此三种 split 的总体数值不能直接作为 sampler 的因果比较。
- 固定 sampler 在 random split 仍优于 learned：`target_proxy` 的 Test Macro-AUC 最高（0.9417），`target_only` 的 Micro-AUC、Macro-F1 和 Micro-F1 最高（0.9503、0.7875、0.8255），耗时仅约 31–37 秒。
- learned 的测试 Macro-AUC 为 0.9092，约 1386 秒；相比 DFS 的 learned 结果（Macro-AUC 0.8095），random split 提升主要来自 BS 可见性结构，而非训练策略本身。
- random split 的 NS 只有 10 个测试样本，Macro-AUC 均未定义，F1 波动很大；ES 也仅 138 个样本。后续应同时报告分组指标和样本数，避免被 BS 主导的总体指标误导。
- `lambda=0` 下 learned 与 random_iterative10 都执行约 10 步，STOP rate 接近 0%，再次表明当前奖励设置不会主动产生稳定的中等长度轨迹。
