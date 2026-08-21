# docs 目录总结报告

> 生成于 readout 实验回退与超参数调优后。本文件是 docs 目录的索引与精简说明
> （readout/attn 结论为历史记录，相关代码已在提交 `8a1bc79` 回退）。

## 目录结构与定位

| 文档 | 定位 | 精简后行数 |
|---|---|---|
| [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) | **当前实验总结**：超参数调优（gamma/hidden/epochs/max_steps/k_hops）与结构变更（value baseline 删除、sampler res-score、predictor readout v1-rd / v1-ppr 负向回退，均为 3 seed 配对）的全部结论、readout 历史结论与设计分析（附文献）、36 次运行总表、建议基线命令 | 281 |
| [TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md) | 训练性能瓶颈与优化方案（由仓库根目录移入并精简） | 73 |
| [SHS27k_BFS_KHOPS1_REINFORCE_GAMMA1_EPOCH10.md](SHS27k_BFS_KHOPS1_REINFORCE_GAMMA1_EPOCH10.md) | 历史单次实验记录（当前 k_hops 语义，gamma=1.0/10ep） | 45 |
| [SHS27k_BFS_REINFORCE_GAMMA1_EPOCH10.md](SHS27k_BFS_REINFORCE_GAMMA1_EPOCH10.md) | 历史单次实验记录（旧 G0/初始邻居机制，当前 CLI 不可复现） | 39 |

## 核心结论速览（详见 EXPERIMENT_SUMMARY.md）

- **当前最优配置**：`v1 + hidden 128 + gamma 0.9 + 20 epochs + max_steps 5`，
  seed 42：test MacF1 0.5239 / MicF1 0.5758 / MacAUC 0.7553 / MicAUC 0.7734。
- **readout（历史，代码已回退）**：纯 ⊙（attn）明显差；补回 |u−v|、u+v
  （attn2）回升但仍低于 v1；瓶颈修复（attn3/attn4）证实 3d→d 压缩是 attn2
  崩盘主因（MacF1 +0.175），但注意力上下文本身净贡献为负 → v1 保持默认，
  分析见 EXPERIMENT_SUMMARY.md「设计分析」节。
- **超参**：gamma 0.9 > 1.0（v1）；hidden 128 > 256（小数据域降过拟合，
  提速 ~35%）；max_steps 5 > 10（v1 全面受益，提速 ~47%）；k_hops 2 > 1
  （s42：val MacAUC 0.7587 vs 0.7390、test 四项全升，但未超 ms5/h128
  最优，单 seed 待扩）；value baseline 删除（3 seed 配对无显著指标差异，
  省时 ~13%，当前代码已为无 value_head 版本）。
- **纪律**：指标方差大（MacF1 0.36~0.52），结论须多 seed 均值±std；
  推理特征禁止含任何标签信息。
- **sampler 打分 v1-res（3 seed）**：残差式 pair 投影 + LN/Tanh 打分头替换
  pairwise MLP，配对 diff 全正（MacF1 +0.050±0.052，3/3 正）且方差显著更小
  （MacF1 std 0.024 vs 0.075，修复 v1 的 s111 崩溃点 0.358→0.481），为当前
  代码默认架构，见 EXPERIMENT_SUMMARY.md TL;DR 第 8 条。
- **predictor readout v1-rd（3 seed，负向已回退）**：readout 加 `u⊙v` 与
  max-pool（打分头 5h→2h→7）配对 diff 全负（MacF1 −0.034±0.040，3/3 负，
  s123 bestEp 19→6 早峰漂移）——小数据域增大打分头容量与 max 型池化净有害，
  代码已回退，见 TL;DR 第 9 条。
- **res-score × k_hops 2（3 seed，纯净配对）**：与 k1 无显著差异（MacF1
  +0.010±0.021，2/3 正；AUC 三项 ≈0），默认保持 k1；s123 单点 MacF1 0.5516
  为全部实验新高（不判定）；v1 时代"k2 正面"由 epoch 数混杂产生，见 TL;DR
  第 10 条。
- **predictor readout v1-ppr（3 seed，负向已回退）**：借鉴 RISE-DDI 的目标
  对结构掩码池化 + PPR 标量注入（打分头 3h→5h+1）配对 diff 负向（MacF1
  −0.049±0.070，s42 崩溃 −0.144 早峰；AUC 3/3 负），代码已回退，见 TL;DR
  第 11 条——小数据域下 predictor readout 扩容净有害（与 v1-rd 同族结论）。

## 精简说明

- `TRAINING_OPTIMIZATION.md` 由根目录移入 docs（`git mv`，根 `CLAUDE.md`
  链接已同步更新），119 → 73 行：压缩瓶颈/实施顺序的描述文字，保留
  优先级表、profiling 项与"必须保持的不变量"。
- 两份历史实验记录删除完整训练曲线表与冗余限制，95/108 → 45/39 行，
  标注历史性并链接新总结。
- 后续更新：readout 代码回退（`8a1bc79`）后同步修正表述；追加 k_hops=2
  实验（133 → 153 行，16 → 17 次）；追加 value baseline 删除实验并扩至
  3 seed 配对（153 → 176 行，17 → 22 次）。

## 更新约定

- 新实验结论只写进 `EXPERIMENT_SUMMARY.md`（保持单一事实源）；
- 历史文档不再追加新内容，仅保留可复现配置与最佳结果；
- 训练稳定性/泄漏红线以 `TRAINING_OPTIMIZATION.md` 与 `src/CLAUDE.md` 为准。
