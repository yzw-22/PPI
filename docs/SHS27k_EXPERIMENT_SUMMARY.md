# SHS27k 实验与可复现性汇总

## 1. 结论摘要

现有证据不足以证明 learned sampler 在 SHS27k 上稳定优于固定 sampler。BFS 历史单次最好结果曾达到 Test Macro/Micro-AUC `0.7479/0.7877`、Macro/Micro-F1 `0.5465/0.6070`，但相同命令、相同 seed 的两次复现实验均未达到该水平；两次复现的均值在四项测试指标上都低于最新的 `target_only` 基线。

因此，本文将历史最好结果保留为可追溯记录，但不再把它视为已复现的性能优势。当前推荐结论是：

- `target_only` 是默认的稳定、低成本基线；
- learned sampler 仍属于实验性配置，需要完成严格确定性检查和多 seed 配对实验后才能评价其真实收益；
- gamma、complexity penalty、DFS 和 Random split 的现有结果均为单次探索性证据，不能用于单变量因果归因；
- 固定阈值 `0.5` 下的 F1 对训练和校准波动尤其敏感，当前应优先依据多次 AUC 统计判断排序能力。

## 2. 实验范围、环境与指标

### 2.1 当前环境快照

最近的 BFS 复现实验运行于：

- 分支：`ablation`；
- commit：`1a075b5`（完整 SHA：`1a075b57636e3e6f8ccb4c9ab4a6ced7026ec049`）；
- GPU：NVIDIA GeForce RTX 5090，driver `580.82.07`；
- PyTorch：`2.7.0a0+7c8ec84dab.nv25.03`；
- CUDA runtime：`12.8`，cuDNN：`9.8.0`；
- PyG：`2.6.1`。

历史实验没有完整保存 GPU、driver 和 Python 依赖快照，因此历史耗时与最近耗时不能严格横向比较。最近复现时工作区相对 commit 仅删除了 `run.sh` 中的 learned 命令，`src/` 训练代码未修改。

### 2.2 统一配置与分组

除特别说明外，实验使用 10 epochs、CUDA、batch size `32`、eval batch size `64`、hidden size `256`、2 层 GAT、4 heads、dropout `0.1`、`fixed_num=1`、`max_steps=10`，Sampler/Predictor 学习率为 `1e-4/1e-3`。checkpoint 按验证集 Macro-AUC 选择，F1 使用固定阈值 `0.5`。

ES/NS/BS 表示测试 PPI 两端相对于训练节点集合的可见性：ES 为一端可见、NS 为两端不可见、BS 为两端均可见。BFS 测试集共 1538 个 PPI，其中 ES=1078、NS=460、BS=0。

当前 learned sampler 使用相邻图增量奖励：

\[
r_t=L_{t-1}-L_t-\lambda\Delta n_t,
\]

其中 `Δn_t` 是本步新增的非 baseline 节点数，DONE/STOP 奖励为 `0`。main 旧实现使用相对 `G_0` 的奖励且没有显式 STOP，因此 main 与当前分支的结果不是单变量比较。

## 3. BFS 可复现性审计

### 3.1 learned sampler 的三次可用记录

三次记录使用相同的 BFS、learned、10 epochs、seed=42、`reinforce_gamma=1.0`、`complexity_penalty=0` 配置。历史单次最好结果的原始 JSON 已被后续同路径实验覆盖，只剩文档中的汇总数值；因此下面的三次统计仅用于诊断波动，不是严格受控 benchmark。

| 运行 | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 平均步数 | STOP rate | 最终节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 历史单次最好，未稳定复现 | 4 | 0.7528 | 0.7479 | 0.7877 | 0.5465 | 0.6070 | 2024.6 | 9.97 | 0.22% | 未记录 |
| repeat1 | 3 | 0.7372 | 0.7318 | 0.7553 | 0.4966 | 0.5402 | 1387.8 | 9.9744 | 0.197% | 14.0307 |
| repeat2 | 7 | 0.7509 | 0.7360 | 0.7406 | 0.4458 | 0.4639 | 1442.7 | 9.9746 | 0.153% | 14.0313 |

三次可用测试结果的诊断统计如下，标准差为样本标准差：

| 指标 | 均值 ± 标准差 | 范围 |
|---|---:|---:|
| Test Macro-AUC | 0.7386 ± 0.0084 | 0.7318–0.7479 |
| Test Micro-AUC | 0.7612 ± 0.0241 | 0.7406–0.7877 |
| Test Macro-F1 | 0.4963 ± 0.0503 | 0.4458–0.5465 |
| Test Micro-F1 | 0.5370 ± 0.0716 | 0.4639–0.6070 |

历史最好结果位于四项指标范围的上端，尤其 Micro-F1 比 repeat2 高 `0.1431`。两次新复现的轨迹长度、STOP rate 和最终节点数几乎相同，但最佳 epoch 和测试 F1 差异明显，说明主要波动来自 Predictor/REINFORCE 优化和 checkpoint 泛化，而不是扩展步数或最终子图规模。

### 3.2 最近复现与最新固定基线

两次新 learned 复现的均值为：Val Macro-AUC `0.7441`，Test Macro/Micro-AUC `0.7339/0.7479`，Test Macro/Micro-F1 `0.4712/0.5020`，耗时 `1415.3s`，平均步数 `9.9745`，STOP rate `0.175%`。

最新固定 sampler 结果来自同一当前环境下重新运行的 BFS、seed=42 实验：

| sampler | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 平均步数 | STOP rate | 最终节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned，最近两次均值 | — | 0.7441 | 0.7339 | 0.7479 | 0.4712 | 0.5020 | 1415.3 | 9.9745 | 0.175% | 14.0310 |
| target_only | 9 | 0.7389 | **0.7422** | 0.7635 | **0.5181** | 0.5331 | **31.7** | 0.0000 | 0% | 2.0000 |
| target_proxy | 5 | 0.7344 | 0.7289 | 0.7607 | 0.4967 | 0.5462 | 39.6 | 0.0000 | 0% | 2.0905 |
| random_1hop10 | 9 | **0.7542** | **0.7437** | **0.7685** | **0.5242** | **0.5501** | 40.9 | 0.0000 | 0% | 11.6616 |
| random_iterative10 | 9 | 0.7476 | 0.7353 | 0.7590 | 0.4902 | 0.5221 | 145.1 | 9.9871 | 0% | 14.0454 |

粗体只标记固定 sampler 中的最好值，不表示统计显著性。最近两次 learned 均值相对最新 `target_only` 的 Test Macro-AUC、Micro-AUC、Macro-F1、Micro-F1 分别低 `0.0083`、`0.0156`、`0.0468`、`0.0310`。在当前证据下，历史 learned 优势已经消失或反转；同时 learned 耗时约为 `target_only` 的 45 倍。

### 3.3 固定 sampler 与历史记录的漂移

| sampler | Δ Val Macro-AUC | Δ Test Macro-AUC | Δ Test Micro-AUC | Δ Macro-F1 | Δ Micro-F1 | Δ 时间(s) |
|---|---:|---:|---:|---:|---:|---:|
| target_only | -0.0000 | +0.0000 | +0.0000 | -0.0000 | -0.0000 | -3.5 |
| target_proxy | -0.0022 | -0.0137 | -0.0068 | -0.0279 | -0.0133 | -21.1 |
| random_1hop10 | +0.0039 | -0.0019 | +0.0107 | +0.1864 | +0.1211 | -25.7 |
| random_iterative10 | -0.0032 | -0.0017 | +0.0061 | +0.0040 | +0.0218 | -92.0 |

这里的差值定义为“最近复现减历史文档记录”。`target_only` 几乎完全复现，说明数据加载和基本训练流程没有整体失效；`target_proxy` 有中等下降；`random_iterative10` 的 AUC 接近历史值。`random_1hop10` 的 AUC 变化较小，但 F1 大幅提高，表明固定阈值 `0.5` 对预测分数校准非常敏感，不能把 F1 变化等同于排序能力同幅度变化。时间普遍下降，但由于历史环境未完整记录，不能归因于单一优化。

## 4. 历史版本与超参数探索

本节保留已有数值用于追溯。除特别说明外，均为单次运行，不能用于证明稳定优势或单变量因果关系。

### 4.1 main 与当前分支历史单次结果

| 实现/配置 | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| main 旧实现，γ=1.0 | 10 | 0.7413 | 0.7279 | 0.7546 | 0.4780 | 0.5029 | 1188.9 |
| 当前分支历史最好，λ=0、γ=1.0 | 4 | 0.7528 | 0.7479 | 0.7877 | 0.5465 | 0.6070 | 2024.6 |
| 当前分支，λ=0、γ=0.9 | 8 | 0.7501 | 0.7417 | 0.7777 | 0.5446 | 0.5895 | 1885.2 |
| 当前分支，λ=0、γ=0.95 | 9 | 0.7580 | 0.7467 | 0.7664 | 0.5000 | 0.5458 | 1408.2 |

main 使用提交 `a113945`，当前分支同时改变了 reward、STOP、proxy/cache、邻接和 sampler 逻辑。历史最好结果未能稳定复现，因此不能继续用它证明当前分支相对 main 有稳定提升，也不能把差异归因于某一项设计。

### 4.2 gamma 消融

在 λ=0、seed=42 的单次结果中，gamma=0.9 和 0.95 没有一致优于 gamma=1.0。三种 gamma 的平均步数均约为 `9.97–9.98`，STOP rate 均低于 `1.1%`，仅改变折扣因子没有产生稳定的中等长度轨迹。由于 gamma=1.0 本身存在明显复现方差，目前不能根据这些单次结果确定最佳 gamma。

### 4.3 complexity penalty 消融

| λ | seed | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 最佳步数 / STOP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 4 | 0.7528 | 0.7479 | 0.7877 | 0.5465 | 0.6070 | 2024.6 | 9.97 / 0.22% |
| 0.001 | 42 | 10 | 0.7449 | 0.7272 | 0.7516 | 0.4573 | 0.4794 | 1301.2 | 9.07 / 21.2% |
| 0.001 | 111 | 3 | 0.7537 | 0.7416 | 0.7365 | 0.4010 | 0.4380 | 1097.1 | 9.18 / 18.6% |
| 0.001 | 123 | 9 | 0.7385 | 0.7430 | 0.7707 | 0.5177 | 0.5680 | 271.2 | 1.00 / 99.89% |
| 0.0015 | 123 | 6 | 0.7393 | 0.7415 | 0.7732 | 0.4730 | 0.5190 | 244.0 | 1.00 / 99.89% |
| 0.002 | 111 | 8 | 0.7414 | 0.7251 | 0.7603 | 0.4780 | 0.5342 | 247.4 | 1.00 / 99.89% |
| 0.005 | 42 | 4 | 0.7531 | 0.7523 | 0.7704 | 0.4788 | 0.5131 | 284.7 | 1.16 / 99.89% |

这些单次结果表明 complexity penalty 对 STOP 行为的影响非常陡峭：λ=0.001 在不同 seed 下既可能保持约 9 步，也可能退化为约 1 步；已有 λ≥0.0015 结果通常接近全 STOP。由于 λ=0 的参考运行本身不稳定，表中不能用于建立可靠的性能排序，只能确认当前 penalty 尺度没有稳定地产生 3–6 步的中间轨迹。

关键 ES/NS 单次结果如下；BFS 的 BS 为空：

| 配置 | ES Macro/Micro-AUC | NS Macro/Micro-AUC | ES Macro/Micro-F1 | NS Macro/Micro-F1 |
|---|---:|---:|---:|---:|
| λ=0、γ=1.0、seed=42，历史最好 | 0.7472 / 0.7846 | 0.7505 / 0.7952 | 0.5472 / 0.6033 | 0.5382 / 0.6161 |
| λ=0.005、seed=42 | 0.7588 / 0.7813 | 0.7389 / 0.7428 | 0.4971 / 0.5338 | 0.4285 / 0.4616 |
| λ=0.001、seed=111 | 0.7433 / 0.7481 | 0.7394 / 0.7080 | 0.4252 / 0.4631 | 0.3408 / 0.3772 |
| λ=0.001、seed=123 | 0.7605 / 0.7884 | 0.7034 / 0.7259 | 0.5467 / 0.5897 | 0.4435 / 0.5149 |
| λ=0.0015、seed=123 | 0.7501 / 0.7805 | 0.7239 / 0.7550 | 0.4919 / 0.5350 | 0.4244 / 0.4788 |
| λ=0.002、seed=111 | 0.7455 / 0.7804 | 0.6734 / 0.7082 | 0.5061 / 0.5598 | 0.4035 / 0.4691 |

NS 对 seed 和惩罚系数较敏感；BS 为空，无法评价双端节点均见于训练图时的泛化能力。

## 5. DFS 单次探索

DFS 实验均为 10 epochs、CUDA、seed=42、`reinforce_gamma=1.0`、`complexity_penalty=0` 的单次结果。测试集共 1529 个 PPI，其中 ES=1215、NS=314、BS=0。

| sampler | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 平均步数 | STOP rate | 最终节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 10 | 0.8153 | 0.8095 | 0.8250 | 0.5782 | 0.6189 | 1389.8 | 9.96 | 0.72% | 14.02 |
| target_only | 10 | 0.8265 | 0.8212 | 0.8430 | 0.5966 | 0.6608 | 31.8 | 0.00 | 0% | 2.00 |
| target_proxy | 10 | 0.8255 | 0.8211 | 0.8379 | 0.6026 | 0.6534 | 38.4 | 0.00 | 0% | 2.09 |
| random_1hop10 | 9 | 0.8139 | 0.8139 | 0.8230 | 0.5746 | 0.6176 | 42.3 | 0.00 | 0% | 11.68 |
| random_iterative10 | 10 | 0.8169 | 0.8091 | 0.8238 | 0.5795 | 0.6208 | 143.4 | 9.99 | 0% | 14.06 |

| sampler | ES AUC (Macro/Micro) | NS AUC (Macro/Micro) | ES F1 (Macro/Micro) | NS F1 (Macro/Micro) |
|---|---:|---:|---:|---:|
| learned | 0.8244 / 0.8394 | 0.7411 / 0.7565 | 0.6049 / 0.6409 | 0.4589 / 0.5163 |
| target_only | 0.8378 / 0.8588 | 0.7432 / 0.7657 | 0.6214 / 0.6866 | 0.4809 / 0.5467 |
| target_proxy | 0.8359 / 0.8515 | 0.7522 / 0.7731 | 0.6228 / 0.6728 | 0.5134 / 0.5663 |
| random_1hop10 | 0.8270 / 0.8344 | 0.7526 / 0.7700 | 0.5930 / 0.6381 | 0.4876 / 0.5255 |
| random_iterative10 | 0.8223 / 0.8340 | 0.7430 / 0.7761 | 0.5966 / 0.6413 | 0.4995 / 0.5286 |

该单次 DFS 结果中，`target_only` 和 `target_proxy` 的总体指标优于 learned，耗时约为 learned 的 1/44–1/36。learned 与 `random_iterative10` 的图规模接近，但未表现出稳定、足以覆盖计算成本的优势。NS 指标普遍低于 ES，BS 为空。DFS 与 BFS 的 split 组成不同，不能把两者总体指标差异解释为 sampler 改进。

## 6. Random split 单次探索

Random split 实验使用相同的 10 epochs、CUDA、seed=42、`reinforce_gamma=1.0`、`complexity_penalty=0` 配置。测试集共 1525 个 PPI，其中 BS=1377、ES=138、NS=10；NS 样本过少，Macro-AUC 均不可定义。

| sampler | 最佳 epoch | Val Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 | 时间(s) | 平均步数 | STOP rate | 最终节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 10 | 0.9054 | 0.9092 | 0.9173 | 0.7251 | 0.7608 | 1385.9 | 9.98 | 0.04% | 14.06 |
| target_only | 10 | 0.9391 | 0.9413 | 0.9503 | 0.7875 | 0.8255 | 31.3 | 0.00 | 0% | 2.00 |
| target_proxy | 10 | 0.9371 | 0.9417 | 0.9493 | 0.7802 | 0.8240 | 36.6 | 0.00 | 0% | 2.09 |
| random_1hop10 | 10 | 0.9103 | 0.9198 | 0.9314 | 0.7266 | 0.7830 | 43.3 | 0.00 | 0% | 11.70 |
| random_iterative10 | 9 | 0.9097 | 0.9207 | 0.9260 | 0.7190 | 0.7649 | 149.7 | 9.98 | 0% | 14.06 |

| sampler | BS AUC (Macro/Micro) | ES AUC (Macro/Micro) | NS AUC (Macro/Micro) | BS F1 (Macro/Micro) | ES F1 (Macro/Micro) | NS F1 (Macro/Micro) |
|---|---:|---:|---:|---:|---:|---:|
| learned | 0.9150 / 0.9222 | 0.8495 / 0.8707 | — / 0.7081 | 0.7412 / 0.7744 | 0.4414 / 0.6052 | 0.1071 / 0.3750 |
| target_only | 0.9485 / 0.9567 | 0.8473 / 0.8758 | — / 0.7037 | 0.8052 / 0.8437 | 0.5614 / 0.6255 | 0.1524 / 0.3871 |
| target_proxy | 0.9494 / 0.9561 | 0.8537 / 0.8700 | — / 0.5649 | 0.7958 / 0.8397 | 0.5834 / 0.6430 | 0.0816 / 0.2963 |
| random_1hop10 | 0.9263 / 0.9371 | 0.8462 / 0.8726 | — / 0.6881 | 0.7419 / 0.7967 | 0.4670 / 0.6203 | 0.1973 / 0.4444 |
| random_iterative10 | 0.9266 / 0.9310 | 0.8657 / 0.8762 | — / 0.6548 | 0.7302 / 0.7785 | 0.5532 / 0.6032 | 0.0879 / 0.2857 |

Random split 的总体指标明显高于 BFS/DFS，主要因为 BS 占测试集约 90%，不能解释为 sampler 本身更强。该单次结果中固定 sampler 仍优于 learned，且耗时低一个数量级以上。NS 只有 10 个样本，相关 F1 波动很大，只能作为描述性信息。

## 7. 已确认的稳定性问题

1. **CUDA 未启用严格确定性。** 训练入口已设置 Python、NumPy、Torch CPU/CUDA seed，但没有设置 `torch.use_deterministic_algorithms(True)`、cuDNN deterministic 选项和 `CUBLAS_WORKSPACE_CONFIG`；DataLoader 也没有独立 generator。当前还启用了 `torch.set_float32_matmul_precision("high")`。
2. **REINFORCE 方差控制不足。** policy loss 直接使用原始 `return_to_go - value`，没有 advantage 标准化、sampler 梯度裁剪或 entropy regularization。
3. **奖励非平稳。** Predictor 与 sampler 交替更新，sampler 所依赖的 Predictor loss 奖励在训练期间持续变化。
4. **checkpoint 选择敏感。** 验证 Macro-AUC 在 epoch 间波动，单个最高 epoch 可能是偶然高点；相同 seed 的最佳 epoch 已从 3 变化到 7。
5. **F1 校准敏感。** 固定 `0.5` 阈值把预测分数的小幅平移放大成明显 F1 变化；`random_1hop10` 的复现结果已经体现这一点。
6. **STOP 行为不连续。** λ=0 时 learned 几乎执行满 10 步，已有 λ≥0.0015 结果通常接近全 STOP，尚未稳定出现 3–6 步轨迹。
7. **计算成本过高。** learned 需要轨迹内多次 Predictor 前向、动态图构造和 Python frontier 操作，而现有精度收益没有通过重复实验。
8. **评估分组有限。** BFS/DFS 的 BS 为空，Random 的 NS 只有 10 个样本，跨 split 总体指标会受到可见性组成强烈影响。

## 8. 后续工作优先级

1. **先完成严格确定性诊断。** 增加 deterministic 运行模式；同一 commit、环境和 seed 连续运行至少 3 次。使用 `torch.use_deterministic_algorithms(True)` 暴露不支持的 PyG/GAT 算子，并记录所有必要的 fallback。
2. **建立配对多 seed 基准。** learned、`target_only`、`target_proxy` 至少使用 5 个相同 seed，保存每次原始 JSON，报告均值、样本标准差、范围及 learned 相对固定基线的配对差值。
3. **降低 REINFORCE 方差。** 依次评估 batch advantage 标准化、sampler 梯度裁剪和小幅 entropy bonus，并分别做单变量消融。
4. **降低奖励非平稳性。** 先 warm-up Predictor，再冻结 Predictor 训练 sampler，之后采用低频交替更新；同时记录 reward、STOP advantage、value loss 和动作熵。
5. **稳定 checkpoint 与 F1。** 评估 checkpoint averaging/EMA；在验证集校准 F1 threshold，并同时保留固定 `0.5` 阈值结果以保证兼容。
6. **最后再搜索 gamma 和 penalty。** 在上述稳定性工作完成前暂停扩大 gamma/complexity penalty 网格；恢复搜索后使用完全相同的配对 seed，并报告性能、平均步数、STOP rate 和耗时的 Pareto 曲线。

计算优化可与稳定性工作并行，包括缓存 `G0`、归一化 embedding、candidate projection 和 adjacency，增量维护 selected state，批量计算 step graph Predictor 前向，以及减少 Python 动态集合与 tensor 构造。但性能优化不得改变随机数消费顺序或采样语义，除非作为独立版本重新建立基准。

## 9. 当前推荐与追溯信息

- 默认基线：`target_only`。它在最近 BFS 运行中几乎完全复现历史结果，测试性能不低于最近 learned 均值，且耗时约为其 1/45。
- learned 实验默认值：若继续诊断，暂保持 `reinforce_gamma=1.0`、`complexity_penalty=0`，避免在未解决复现问题前引入额外变量；这不是部署推荐。
- 不推荐把 λ=0.005 的短耗时视为有效优化，因为其 sampler 几乎总是 STOP。
- 所有单次最好值都应标注 seed、commit、环境和证据等级，不再用于单独宣称模型改进。

典型命令：

```bash
python -m src.train_shs27k \
  --dataset SHS27k --split <bfs|dfs|random> \
  --sampler-mode <learned|target_only|target_proxy|random_1hop10|random_iterative10> \
  --device cuda --epochs 10 --seed <SEED> \
  --reinforce-gamma <GAMMA> \
  --complexity-penalty <LAMBDA> \
  --output <RESULT.json>
```

最近复现的原始文件位于 `/tmp/ppi_shs27k_*.json`，其中 learned 两次副本为：

- `/tmp/ppi_shs27k_bfs_learned_repeat1.json`；
- `/tmp/ppi_shs27k_bfs_learned_repeat2.json`。

`/tmp` 不是长期归档目录，且相同 `--output` 会覆盖历史结果。后续正式实验应写入版本化结果目录，文件名至少包含 split、sampler、seed、commit 和时间戳，并同时保存命令、环境依赖、GPU/driver 信息及工作区 diff。
