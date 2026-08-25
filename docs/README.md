# PPI 项目报告（docs 唯一文档）

> 本文件是 `docs/` 目录的唯一报告，替代原 README 索引、TRAINING_OPTIMIZATION、
> EXPERIMENT_SUMMARY 与两份历史实验记录。
>
> 背景：训练/验证/测试的采样知识图谱（KG）由 split-local 图改为**全数据集全图**
> （见下文「知识图谱设计」），原文档中所有以 split-local KG 为前提的表述与实验
> 结论不再直接成立，故压缩为单份报告。本文同时是当前代码的唯一文档事实源。

## 1. 项目概述

- 7 类 PPI 多标签预测（reaction / binding / ptmod / activation / inhibition /
  catalysis / expression），标签为按无向蛋白对 OR 聚合的 multi-hot 向量。
- 蛋白特征为预计算 ESM-2 3B（`esm2_t36_3B_UR50D`）平均池化 embedding，
  `[M, 2560]` bfloat16（`{name}_tensor.pt`，`tensor[i]` = Protein Index `i`）。
- 数据：SHS27k（1690 蛋白 / 7624 PPI）、SHS148k（5189 / 44488）、
  STRING（15335 / 593397）；`ppi_list` 中每个无向 PPI 对恰好出现一次（方向任意，
  不存在双向重复），`ppi_list` 行号 = PPI Index。
- 管线：split 选出目标 PPI 对 → 全图 KG 上的 REINFORCE 子图 Sampler → GAT
  Predictor（多图 batch 编码 → `h_u+h_v`、`|h_u−h_v|`、图均值 → 7 维 logits）。

## 2. 知识图谱设计（当前）

- **训练、验证、测试三阶段共用同一张全图**：节点 = 全部蛋白，边 = 全部 PPI 对
  （双向），特征 = 全部 ESM embedding，proxy 候选池 = 全部蛋白。由
  `PPIGraph.build_full_graph(undirected=True)` 构建，`train_shs27k.py` 中只构建
  一次并全程共享（含邻接表 `full_adjacency`）。
- **split 只做两件事**：① 决定各阶段的目标 PPI 对（`get_ppi_indices` 提取
  train/val/test targets 与 labels）；② 提供训练节点集合，用于测试集
  BS/ES/NS 可见性分组（`train_node_index`，来自 `build_graph("train")`）。
- **不变量（必须保持）**：
  - 目标边 `(u,v)` 与 `(v,u)` 在每次采样前从安全邻接中排除（
    `_TargetSafeAdjacency` 惰性视图，共享邻接不被修改），且不进入 `G_0` 与
    step graph；
  - `G_0`（`baseline_graph`）只含 `u`、`v` 与必要虚拟 proxy（安全邻接为空的
    目标才选 proxy，按 embedding 余弦相似度，两目标可共享）；`G_0` 保留选中
    节点间的全部安全诱导边；
  - 候选限制在距 `G_0` 种子不超过 `k_hops`（默认 1）的安全区域内；`max_steps`
    （默认 10）只限制动作次数，无 STOP 动作；
  - Predictor 训练与评估均使用 `final_graph`（无动作时即 `G_0`）；
  - **推理特征只含 ESM embedding 与无标签拓扑**：`edge_label` 是公共返回字段
    但训练流程从不读取，标签只作为 train/val/test 目标对的 BCE 目标——全图
    含 val/test 边作为拓扑不构成标签泄漏；
  - 奖励/return-to-go/advantage 保持 FP32；F1 阈值固定 0.5。
- 说明：`build_graph(split_name)` 的 split-local 构建保留（节点集合查询、测试、
  外部使用），但训练入口不再以它作为 KG。

## 3. 训练协议（AlternatingTrainer）

每 batch 交替两次更新：

1. **Sampler 更新**（Predictor 冻结，stochastic 轨迹）：对每条轨迹计算
   `G_0` 与所有 step graph 的 BCE loss，增量奖励 `r_t = L(G_{t-1}) − L(G_t)`
   （第一步以 `G_0` loss 为前项；无子图大小/Δn 惩罚），按 `G_t = r_t + γG_{t+1}`
   计算 return-to-go；无学习 baseline，advantage 即 detached RTG，batch 内
   标准化 `Â = (A − mean)/max(std, 1e-8)` 后算
   `L_policy = −Σ log π(a_t|s_t)·stopgrad(Â)` 更新 Sampler。
2. **Predictor 更新**（Sampler 冻结，greedy 轨迹）：只用每条轨迹的
   `final_graph` 做 BCE with logits 更新。

动作打分：state/candidate 独立投影，拼接映射回 hidden_dim 并加回投影 state
（残差）→ `Linear(d→d//2) → LayerNorm → Tanh → Linear(d//2→1)` → softmax；
训练记录 Categorical `log_prob`，评估/更新用 greedy argmax。

超参默认：`--epochs 10 --batch-size 32 --eval-batch-size 64 --hidden-dim 256
--max-steps 10 --k-hops 1 --gnn-layers 2 --heads 4 --dropout 0.1
--sampler-lr 1e-4 --predictor-lr 1e-3 --reinforce-gamma 1.0`。
历史推荐配置（split-local 时代，见 §6）：`h128 / γ0.9 / 20ep / ms5`。

## 4. 评估协议

- 训练期间**只评估验证集**（每 epoch 一次）；按验证 Macro-AUC 保存最佳
  Sampler/Predictor 状态（`--checkpoint-dir` 指定时落盘，否则内存保存）；
  训练结束后**只在最佳 checkpoint 上测试一次**。
- 指标：Macro/Micro ROC-AUC 与固定阈值 0.5 的 Macro/Micro F1；常数类（单值）
  子集 AUC 报 `None`（与空分组同约定），不参与最佳选择。
- 测试集按两端节点相对训练节点集合的可见性分 BS（两端均训练可见）/ ES（单端）
  / NS（均不可见）；空分组返回 `count=0` 与 `None`。
- 可见性分组**仍按 train split 节点集**计算（全图下"图中节点"无区分度）。

## 5. 数据与性能

### 5.1 规模（全图即有向边数）

| 数据集 | 蛋白 | PPI 对 | 全图有向边 | 全图 node_feat(fp32) |
|---|---:|---:|---:|---:|
| SHS27k | 1,690 | 7,624 | 15,248 | ~17 MB |
| SHS148k | 5,189 | 44,488 | 88,976 | ~53 MB |
| STRING | 15,335 | 593,397 | 1,186,794 | ~157 MB |

每个蛋白都出现在 ≥1 条 PPI 中，因此全图 `node_index` 覆盖全部蛋白且局部 id =
全局 id（实现仍用 `unique()` 通用处理）。

### 5.2 数据喂入（结论）

训练循环用 `torch.randperm` + 批量 fancy-index gather 直接从预载 tensor 取数
（`train_targets[batch_indices]` / `train_labels[batch_indices]`），这是该负载
（数据完全驻留内存/显存、无 I/O、无逐样本预处理）下的最优形态：
实测（SHS27k train）每 epoch 取数 0.75 ms；DataLoader `num_workers=0` 慢 ~34×，
`num_workers=2` 因每 epoch 重 spawn + pickle 整个 PPIGraph 慢 ~3800×，且 CUDA
下 `get_dataloader` 的守卫禁止 workers/pin_memory、逐样本 `int()` 会引入 D2H
同步。`get_dataloader`/`_PPIDataset` 保留为公共 API（测试/外部轻量使用），
训练路径不经过它。

### 5.3 邻接与计算

- 全图邻接（不可变 `tuple[frozenset]`）**一次性构建**，训练 batch、每 epoch
  验证、最终测试共享（`train_shs27k.py` 与 `evaluate(..., adjacency=...)`）；
  目标边由惰性视图逐 target 排除，无逐 target 复制。
- 采样区域/候选池随全图度数扩大（STRING 的 hub 目标 1-hop 区域可能很大），
  受 `k_hops`/`max_steps` 约束；训练时间主要由 Sampler 轨迹生成、Sampler 更新
  阶段对多个 step graph 的 Predictor 前向、验证推理构成。
- 仍有效的优化方向（与 split-local 无关的部分）：P1 缓存归一化 embedding 与
  candidate 投影、增量维护 selected embedding 均值、减少/合并 Predictor 重复
  forward；P2 评估 `torch.inference_mode()`、batch size 扫描、Predictor BF16/
  AMP；P3 STRING 用 CSR tensor 邻接 + tensor frontier（全图下收益更大）。
  Profiling 需分别统计 sampler_g0_build / sampler_action_score /
  sampler_graph_build / predictor_baseline / predictor_step_graphs /
  predictor_update / validation / test_evaluation；CUDA 计时前
  `torch.cuda.synchronize()`。

## 6. 历史实验记录（split-local KG 时代，仅作配置起点）

> **失效声明**：以下结论全部产生于 KG 为 split-local 图的旧代码（全图改造
> 前），数值与排名不可直接迁移到全图 KG，仅保留为超参/协议起点与仓库历史。
> 全图改造后需重新验证。

- 56 次运行（3-seed 配对为主）关键结论：
  - 最优配置（旧）：`res-score + hidden 128 + gamma 0.9 + 20 epochs +
    max_steps 5`（SHS27k bfs test MacF1 0.5239）；res-score 打分头为当前代码
    默认架构（配对 MacF1 +0.050±0.052，方差更小）；value baseline 删除无显著
    指标差异（省时 ~13%），当前代码即无 value_head 版本。
  - 划分效应是最大"杠杆"且**跨划分不可混比**：dfs 划分任务系统性更易
    （SHS27k MacF1 0.601 vs bfs 0.515，测试可见性 79.5% vs 70.1%）；新随机
    bfs 划分波动即达 MacF1 ~0.03-0.06；SHS148k random 指标虚高（test BS
    96.5%，预测退化为已见节点模式匹配）；SHS148k bfs 训练崩溃为划分数据属性
    （train 稠密核心 vs val 稀疏外围），与 RL 机制无关。
  - sampler/RL 在 dfs 划分上无正向贡献（ms0 即达全配水平：SHS27k dfs MacF1
    0.6295±0.0172 vs ms5 0.6006，3/3 正；SHS148k dfs 持平但省时 ~14×）；
    G0-only predictor 为旧代码确认的最优形态。
  - 长训练：40ep 为成本-收益甜点（SHS148k dfs ms0 valMacAUC 0.8557±0.0004），
    h256 vs h128 无显著差异；k_hops 2 纯净配对无显著收益，默认 k1。
- 纪律（仍然适用）：指标方差大（MacF1 0.36~0.52），结论须多 seed 配对均值±std；
  推理特征禁止任何标签信息；报告结果须标注数据集与 split。
- 建议命令（配置起点，全图 KG 自动生效）：

```bash
python -m src.train_shs27k \
  --dataset SHS27k --split bfs --device cuda \
  --epochs 20 --hidden-dim 128 --max-steps 5 --reinforce-gamma 0.9 \
  --seed 42 --output /tmp/ppi_best.json
```

## 7. 全图 KG 实验记录（SHS27k bfs 3-seed 基线，2025-08-24）

配置与 §6 旧基线同款（`ms5/h128/g0.9/20ep`，seed 42/111/123，CPU），唯一差别是
KG 由 split-local 改为全图（提交 `23794a8` 之后）。原始 JSON：
`/tmp/ppi_fullkg_s{42,111,123}.json`。

| seed | 秒 | bestEp | valMacAUC | MacAUC | MicAUC | MacF1 | MicF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1221 | 9 | 0.7717 | 0.7817 | 0.7981 | 0.5789 | 0.5974 |
| 111 | 1148 | 6 | 0.7574 | 0.7539 | 0.7739 | 0.4876 | 0.5537 |
| 123 | 1037 | 14 | 0.7867 | 0.7917 | 0.8183 | 0.5879 | 0.6285 |
| 均值±std | 1136±93 | — | 0.7719±0.0146 | 0.7758±0.0196 | 0.7968±0.0223 | 0.5515±0.0555 | 0.5932±0.0376 |

与 split-local 同配置 3-seed 配对 diff（全图 − 旧，五项指标全为 3/3 正）：

| 指标 | 配对 diff | 旧基线均值 |
|---|---|---:|
| MacAUC | +0.0248±0.0120 | 0.7509±0.0075 |
| MicAUC | +0.0226±0.0124 | 0.7742±0.0161 |
| MacF1 | +0.0367±0.0266 | 0.5148±0.0238 |
| MicF1 | +0.0510±0.0144 | 0.5422±0.0379 |
| valMacAUC | +0.0155±0.0056 | 0.7564±0.0116 |

- 速度：epoch ~52-65 s，总时长 1037-1221 s（s42 1221 s vs 旧 1179 s 持平）；
  全图 KG 在 SHS27k 上无速度损失（邻域扩大被小图规模抵消）。
- s111 仍为最弱 seed（MacF1 0.4876 vs 旧 0.4811，全图下相对差距略放大），
  MacF1 的 std 由 0.024 增至 0.055，方差主要来自 s111。
- 测试可见性分组计数不变（BS 0 / ES 1078 / NS 460），符合预期（分组与 KG 无关）。
- 单配置 3-seed 配对，方向一致但样本小，结论待更多 seed 验证。

## 8. 全图 KG 实验记录（SHS27k bfs max_steps 消融：ms0 / ms10，2025-08-25）

### 8.0 ms0（`--max-steps 0`，G0-only）

配置同 §7（`h128/g0.9/20ep`，seed 42/111/123，CPU），仅 `--max-steps 0`：
轨迹无动作、`final_graph` ≡ `G_0`，Sampler 不更新（`sampler_loss`/`mean_reward`
恒为 0），Predictor 只在 G0 上训练。3 seed 并行（每进程 3 线程，总墙钟 ~87 s）。
原始 JSON：`/tmp/ppi_ms0_s{42,111,123}.json`。

| seed | 秒 | bestEp | valMacAUC | MacAUC | MicAUC | MacF1 | MicF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 87 | 14 | 0.7436 | 0.7360 | 0.7427 | 0.4935 | 0.5189 |
| 111 | 87 | 12 | 0.7555 | 0.7574 | 0.7709 | 0.4943 | 0.5510 |
| 123 | 87 | 20 | 0.7595 | 0.7558 | 0.7801 | 0.5256 | 0.5789 |
| 均值±std | 87.0±0.4 | — | 0.7529±0.0083 | 0.7497±0.0120 | 0.7646±0.0195 | 0.5045±0.0183 | 0.5496±0.0300 |

与 §7 ms5 基线配对 diff（ms0 − ms5）：

| 指标 | 配对 diff | 正向 seed 数 |
|---|---|---:|
| valMacAUC | −0.0191±0.0149 | 0/3 |
| MacAUC | −0.0260±0.0261 | 1/3 |
| MicAUC | −0.0322±0.0267 | 0/3 |
| MacF1 | −0.0470±0.0479 | 1/3 |
| MicF1 | −0.0436±0.0382 | 0/3 |
| 秒 | −1048±92 | — |

- 速度：epoch ~4.3 s（3 线程并行），总 87 s vs ms5 1136 s，**约 13×**（省去
  Sampler 更新与 step-graph 前向，这两者是 ms5 的训练耗时主体）。
- 指标：ms0 五项均值全面低于 ms5（val/Mic 0/3 正、Mac 1/3 正），差距集中在
  s42/s123（MacF1 −0.085/−0.062）；s111 例外（MacAUC +0.0035、MacF1 +0.0067，
  基本持平）。**全图 KG + SHS27k bfs 下采样动作（ms5）有正向贡献**，与 §6
  split-local 时代"dfs 上 ms0 最优"不冲突（划分/KG 均不同）；但差距与 seed 间
  方差同量级（MacF1 差 ~0.05 vs std ~0.05），仅 3-seed，待扩 seed 确认。
- bestEp 后移（12-20 vs ms5 的 6-14）：无采样时验证曲线峰值更晚。
- 可见性分组计数不变（BS 0 / ES 1078 / NS 460）。

### 8.1 ms10（`--max-steps 10`）

同批补跑（同配置 `h128/g0.9/20ep`，seed 42/111/123，CPU），3 seed 并行
（每进程 3 线程）。原始 JSON：`/tmp/ppi_ms10_s{42,111,123}.json`。

| seed | 秒 | bestEp | valMacAUC | MacAUC | MicAUC | MacF1 | MicF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2929 | 8 | 0.7522 | 0.7560 | 0.7820 | 0.5499 | 0.5751 |
| 111 | 3024 | 15 | 0.7728 | 0.7777 | 0.7745 | 0.5242 | 0.5434 |
| 123 | 2897 | 8 | 0.7546 | 0.7667 | 0.7929 | 0.5534 | 0.6097 |
| 均值±std | 2950±66 | — | 0.7599±0.0113 | 0.7668±0.0109 | 0.7831±0.0092 | 0.5425±0.0160 | 0.5761±0.0332 |

配对 diff（括号内为正向 seed 数）：

| 对比 | valMacAUC | MacAUC | MicAUC | MacF1 | MicF1 | 秒 |
|---|---|---|---|---|---|---:|
| ms10 − ms5 | −0.0121±0.0246（1/3） | −0.0090±0.0284（1/3） | −0.0136±0.0132（1/3） | −0.0090±0.0395（1/3） | −0.0171±0.0062（0/3） | +1815±93 |
| ms10 − ms0 | +0.0070±0.0112（2/3） | +0.0171±0.0054（3/3） | +0.0186±0.0185（3/3） | +0.0380±0.0160（3/3） | +0.0265±0.0321（2/3） | +2863±66 |

- 排序（均值）：五项指标均为 **ms5 > ms10 > ms0**；ms10−ms5 差距小且在 std
  内（仅 1/3 正），ms10−ms0 差距大（MacAUC/MicAUC/MacF1 3/3 正）——采样动作的
  贡献稳健，最优步数在 ms5 附近（5-10 间或为平台期），3-seed 不足以区分
  ms5/ms10。
- s111（ms5 最弱 seed）在 ms10 下明显改善（MacAUC +0.0238、MacF1 +0.0366），
  MacF1 的 seed 间方差由 ms5 的 0.055 降至 0.016——更长步数上限对弱 seed 更稳。
- 速度：ms10 epoch ~147 s（3 线程并行）；与 ms5（~55 s/epoch @ 8 线程）按线程
  数折算后单 epoch 计算量同量级——`k_hops=1` 下 frontier 常在 5 步内耗尽，10
  步上限很少真正触达；总墙钟 2950 s 主要是"并行 3 线程 vs 顺序 8 线程"的
  差异，不是纯 max_steps 效应。
- 可见性分组计数不变（BS 0 / ES 1078 / NS 460）。

### 8.2 static（`--sampler static`：G0 + 全部 1-hop 安全邻居，不可学习）

配置同 §7（`h128/g0.9/20ep`，seed 42/111/123，CPU），仅换用
`StaticNeighborhoodSampler`（提交 `425ef43` 引入）：图 = G0 种子（u/v/必要
proxy）的全部安全 `k_hops=1` 区域诱导子图，无动作、零参数，训练只更新
Predictor。3 seed 并行（每进程 3 线程）。原始 JSON：
`/tmp/ppi_static_s{42,111,123}.json`。

| seed | 秒 | bestEp | valMacAUC | MacAUC | MicAUC | MacF1 | MicF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 338 | 18 | 0.8022 | 0.7995 | 0.8189 | 0.5961 | 0.6210 |
| 111 | 336 | 11 | 0.8068 | 0.8085 | 0.8233 | 0.6063 | 0.6280 |
| 123 | 336 | 17 | 0.7992 | 0.8061 | 0.8187 | 0.5819 | 0.6064 |
| 均值±std | 337±1 | — | 0.8027±0.0038 | 0.8047±0.0046 | 0.8203±0.0026 | 0.5947±0.0123 | 0.6185±0.0110 |

配对 diff（括号内为正向 seed 数）：

| 对比 | valMacAUC | MacAUC | MicAUC | MacF1 | MicF1 | 秒 |
|---|---|---|---|---|---|---:|
| static − ms5 | +0.0308±0.0185（3/3） | +0.0289±0.0223（3/3） | +0.0235±0.0246（3/3） | +0.0433±0.0663（2/3） | +0.0253±0.0482（2/3） | −799±92 |
| static − ms0 | +0.0498±0.0095（3/3） | +0.0550±0.0074（3/3） | +0.0557±0.0190（3/3） | +0.0903±0.0298（3/3） | +0.0689±0.0380（3/3） | +250±1 |

图规模（train 目标 4562 个实测）：静态图节点 mean 45.6 / median 41 / max 156，
无向边 mean 238 / median 131 / max 1451（hub 目标区域大）；对照 G0 恒为 2 节点
（全图下安全邻接非空、无 proxy）。

- **结论：RL 选取在 SHS27k bfs 全图 KG 上无正向作用，且显著劣于全 1-hop 邻域**：
  static 的 AUC 三项 3/3 正、F1 2/3 正（MacAUC 均值 +0.029、valMacAUC +0.031、
  MacF1 +0.043）；s111（ms5 最弱 seed）提升最大（MacAUC +0.055、MacF1 +0.119）；
  方差大幅下降（MacAUC std 0.0196→0.0046，MacF1 std 0.0555→0.0123）。
- 机制解释：`k_hops=1` 候选区域平均 ~46 节点（RL 每轨迹最多选 5-10 个），
  static 全取；GAT Predictor 从完整邻域获得更多上下文，学习式截断反而丢信息，
  在该数据上未见"降噪"收益。
- 速度：static 每 epoch ~16.7 s（3 线程），总 337 s ≈ ms5 的 1/3.4（省去 sampler
  更新与 step-graph 前向；Predictor 前向图更大但总量更小）。
- 三档排序（均值，五项指标）：**static > ms5 > ms0**——采样器在 SHS27k bfs
  全图 KG 上的最优形态是"全取 1-hop 邻域"，RL 选取与 G0-only 均非最优。
  与 §6 split-local 时代"G0-only 最优（dfs）"不冲突（划分/KG 均不同），但本次
  结果提示：该数据下 RL 选取机制可被静态邻域替代。
- 若需进一步定位"选取策略"与"上下文大小"的效应，可加"随机选取同等规模子集"
  对照（后续可选）。
- 可见性分组计数不变（BS 0 / ES 1078 / NS 460）。

## 9. 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```
