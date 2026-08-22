# docs 目录总结报告

> 生成于超参数调优与结构变更实验后。本文件是 docs 目录的索引与精简说明。

## 目录结构与定位

| 文档 | 定位 | 精简后行数 |
|---|---|---|
| [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) | **当前实验总结**：超参数调优（gamma/hidden/epochs/max_steps/k_hops）、结构变更（value baseline 删除、sampler res-score）、划分对比（bfs/dfs/random、SHS27k/SHS148k）、G0-only 诊断（崩溃根因定位 + sampler 冗余/有害判定）、dfs 长训练与 random 划分可见性机制的全部结论（均为 3 seed 配对）、56 次运行总表、建议基线命令；已被多 seed 否决的设计历史（attn 系列 readout、u⊙v+max-pool、掩码池化+PPR）已删除，详见 git 历史 `8a1bc79`/`6bbfaca`/`0d4f354` | 380 |
| [TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md) | 训练性能瓶颈与优化方案（由仓库根目录移入并精简） | 73 |
| [SHS27k_BFS_KHOPS1_REINFORCE_GAMMA1_EPOCH10.md](SHS27k_BFS_KHOPS1_REINFORCE_GAMMA1_EPOCH10.md) | 历史单次实验记录（当前 k_hops 语义，gamma=1.0/10ep） | 45 |
| [SHS27k_BFS_REINFORCE_GAMMA1_EPOCH10.md](SHS27k_BFS_REINFORCE_GAMMA1_EPOCH10.md) | 历史单次实验记录（旧 G0/初始邻居机制，当前 CLI 不可复现） | 39 |

## 核心结论速览（详见 EXPERIMENT_SUMMARY.md）

- **当前最优配置**：`v1 + hidden 128 + gamma 0.9 + 20 epochs + max_steps 5`，
  seed 42：test MacF1 0.5239 / MicF1 0.5758 / MacAUC 0.7553 / MicAUC 0.7734。
- **readout 设计**：当前为 v1 直通（`h_u+h_v`、`|h_u−h_v|`、图均值 → 3h→7）。
  曾试的注意力 readout、`u⊙v`+max-pool、掩码池化+PPR 等扩容方案均已多 seed
  否决并回退（详见 git 历史），文档不再保留其分析。
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
  代码默认架构，见 EXPERIMENT_SUMMARY.md TL;DR 第 7 条。
- **res-score × k_hops 2（3 seed，纯净配对）**：与 k1 无显著差异（MacF1
  +0.010±0.021，2/3 正；AUC 三项 ≈0），默认保持 k1；s123 单点 MacF1 0.5516
  为全部实验新高（不判定）；v1 时代"k2 正面"由 epoch 数混杂产生，见 TL;DR
  第 8 条。
- **SHS27k dfs 划分（3 seed，首次 dfs）**：同配置（ms5/h128/g0.9/20ep）
  下五项指标 3/3 大幅优于 bfs（MacF1 0.6006±0.0071 vs 0.5148±0.0238，
  MacAUC 0.8186 vs 0.7509），方差仅为 bfs 约 1/4——但这是划分难度属性
  （dfs 测试集训练可见占比 79.5% vs 70.1%），非模型改进，跨划分不可混比，
  见 TL;DR 第 9 条。
- **SHS148k dfs 划分（3 seed，首次 SHS148k）**：同配置下 MacF1
  0.6355±0.0139 / MacAUC 0.8501（bestEp 全 19，20ep 可能未到上限），较
  SHS27k dfs 五项 3/3 正差（MacF1 +0.035）——归因数据规模（train 5.8×）
  与测试可见性（86.1%），跨数据集不可配对，非模型改进；单次训练 ~4.1 h，
  见 TL;DR 第 10 条。
- **新随机 bfs 划分（3 seed，`dataset_ppisplit/`）**：SHS27k 新 bfs 全面
  优于旧 bfs（MacF1 0.5479±0.0158 vs 0.5148，3/3 正）——划分随机波动本身
  达 MacF1 ~0.03-0.06 量级，见 TL;DR 第 11 条；SHS148k bfs 则灾难性差于
  dfs（MacF1 0.4163 vs 0.6355，0/3 正，s42/s123 崩溃 bestEp=1 早峰），
  bfs 划分效应在更大数据上被放大，见 TL;DR 第 12 条。
- **G0-only 诊断（max_steps=0，无子图/无 RL）**：SHS148k bfs 的"val 不随
  训练改善"在去掉子图与 REINFORCE 后依然存在（valMacAUC 0.7327±0.0025，
  3 seed，testMacF1 0.4282±0.0153）——崩溃根因是 bfs 划分数据属性（train
  稠密核心 vs val 稀疏外围），非 RL 机制；dfs 对照 G0-only 即达全配水平
  （0.8468 vs 0.8476）——dfs 上子图通路贡献≈0，高指标来自 pair 特征本身，
  sampler 在 dfs 上近乎冗余（3 seed 确认，省时 ~14×），见 TL;DR 第 13/14 条。
- **长训练/hidden（ms0 形态）**：40ep 为甜点（SHS148k dfs 3-seed
  valMacAUC 0.8557±0.0004 vs 20ep 0.8482，MacF1 持平、方差小 16×），60ep
  无额外收益，h256 无显著差异（s42 单点 +0.014 被 s111 −0.021 抵消）；
  SHS27k dfs 上 ms0 亦 ≥ ms5（0.6129 vs 0.6008，单 seed），见 TL;DR 第 15 条。
- **SHS148k random 划分（3 seed，首次 random）**：MacF1 0.8444±0.0109 /
  valMacAUC 0.9660±0.0001——但为**划分可见性虚高**（test BS 96.5%，train/
  test 节点高度共享，预测退化为已见节点模式匹配），非模型改进，与 dfs
  （BS 0）不可混比；报告须标注划分，见 TL;DR 第 16 条。
- **ms0 提升（SHS27k dfs 3-seed 纯净配对，见 TL;DR 第 17 条）**：去掉
  sampler 后 MacF1 0.6006±0.0071 → **0.6295±0.0172（配对 +0.029±0.022，
  3/3 正）**、valMacAUC +0.019（3/3 正）——小数据域下 RL 采样净有害；
  SHS148k dfs ms0 持平（0.6355→0.6352）但成本 ↓14×。**sampler/RL 在两个
  dfs 数据集均无正向贡献，G0-only predictor 为确认最优形态。**

## 精简说明

- `TRAINING_OPTIMIZATION.md` 由根目录移入 docs（`git mv`，根 `CLAUDE.md`
  链接已同步更新），119 → 73 行：压缩瓶颈/实施顺序的描述文字，保留
  优先级表、profiling 项与"必须保持的不变量"。
- 两份历史实验记录删除完整训练曲线表与冗余限制，95/108 → 45/39 行，
  标注历史性并链接新总结。
- 后续更新：readout 代码回退（`8a1bc79`）后同步修正表述；追加 k_hops=2
  实验（133 → 153 行，16 → 17 次）；追加 value baseline 删除实验并扩至 3 seed 配对（153 → 176 行，
  17 → 22 次）；划分实验（dfs/SHS148k/新 bfs，176 → 401 行，22 → 48 次）；
  删去多 seed 否决的设计历史（attn 系列/u⊙v+max-pool/PPR，401 → 278 行，
  48 → 32 次）；追加 SHS148k G0-only 崩溃诊断（278 → 315 行，32 → 36 次）；
  追加 dfs ms0 3-seed 等价性与 40ep 长训练探测（315 → 329 行，36 → 39 次）；
  追加 40ep 3-seed/h256/60ep 探测（329 → 349 行，39 → 46 次）；
  追加 random 划分发现与 SHS27k dfs ms0 验证（349 → 363 行，46 → 50 次）；
  追加 SHS27k dfs ms0 3-seed 提升确认（363 → 380 行，50 → 56 次）。

## 更新约定

- 新实验结论只写进 `EXPERIMENT_SUMMARY.md`（保持单一事实源）；
- 历史文档不再追加新内容，仅保留可复现配置与最佳结果；
- 训练稳定性/泄漏红线以 `TRAINING_OPTIMIZATION.md` 与 `src/CLAUDE.md` 为准。
