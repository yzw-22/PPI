# PPI 项目报告（docs 唯一文档）

> 本文件是 `docs/` 目录的唯一报告，替代原 README 索引、TRAINING_OPTIMIZATION、
> EXPERIMENT_SUMMARY 与两份历史实验记录。
>
> 2026-09-05 精简：实验记录压缩为「结果总表 + 主要结论」（§6），逐 seed 明细表与
> 次要配对 diff 已删除。原始逐 seed JSON（`/tmp/ppi_*.json` 等）为易失数据，本文档
> 为持久记录，均值±std 即为准数。
>
> 背景：训练/验证/测试的采样知识图谱（KG）由 split-local 图改为**全数据集全图**
> （提交 `23794a8`，见 §2）；split-local 时代的结论仅作历史背景保留（§6.3）。

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
- 可选 `--use-edge-relations` 将 train split 边的 7 维 multi-hot relation 输入
  Predictor GAT；val/test 边仍保留为无标签拓扑，但 relation 全零，虚拟 proxy
  边与 self-loop 也为全零。该开关默认关闭，以保持历史实验可复现。
- 可选 `--use-sampler-edge-relations` 独立将 train-edge relation 输入 RL 动作
  打分：候选节点连接当前已选子图的关系按逐维 max/OR 聚合、投影后加到 candidate
  表示。目标边先从安全邻接删除，held-out 与 proxy 边 relation 为零。
- **不变量（必须保持）**：
  - 目标边 `(u,v)` 与 `(v,u)` 在每次采样前从安全邻接中排除（
    `_TargetSafeAdjacency` 惰性视图，共享邻接不被修改），且不进入 `G_0` 与
    step graph；
  - `G_0`（`baseline_graph`）只含 `u`、`v` 与必要虚拟 proxy（安全邻接为空的
    目标才选 proxy，按 embedding 余弦相似度，两目标可共享）；`G_0` 保留选中
    节点间的全部安全诱导边；
  - 候选限制在距 `G_0` 种子不超过 `k_hops`（默认 1）的安全区域内；`max_steps`
    （默认 10）只限制动作次数，无 STOP 动作；
  - Predictor 训练与评估均使用 `prediction_graph`（末步图；无动作时为 A2
    静态底座 `reference_graph`，再退回 `G_0`）；
  - 默认模式的推理特征只含 ESM embedding 与无标签拓扑；relation-aware 模式
    额外读取 train split 边标签。val/test relation 从不进入关系查询表，且当前
    查询目标边在 edge feature 物化前已删除；
  - 奖励/return-to-go/advantage 保持 FP32；F1 阈值固定 0.5。
- 说明：`build_graph(split_name)` 的 split-local 构建保留（节点集合查询、测试、
  外部使用），但训练入口不再以它作为 KG。

## 3. 训练协议（AlternatingTrainer）

每 batch 交替两次更新：

1. **Sampler 更新**（Predictor 冻结，stochastic 轨迹）：对每条轨迹计算
   `G_0` 与所有 step graph 的 BCE loss，增量奖励 `r_t = L(G_{t-1}) − L(G_t)`
   （第一步以 `G_0` loss 为前项；无子图大小/Δn 惩罚）；`--reward-margin` 改为
   固定参照图的边际改进（`M(p)=mean_j((2y_j−1)·p_j)`，非对称缩放默认 2:1，
   参照可选 G0 或 A2 静态底座）。按 `G_t = r_t + γG_{t+1}` 计算 return-to-go；
   无学习 baseline，advantage 即 detached RTG，batch 内标准化
   `Â = (A − mean)/max(std, 1e-8)` 后算
   `L_policy = −Σ log π(a_t|s_t)·stopgrad(Â)` 更新 Sampler。
2. **Predictor 更新**（Sampler 冻结，greedy 轨迹）：只用每条轨迹的
   `prediction_graph` 做 BCE with logits 更新。

动作打分：state/candidate 独立投影，拼接映射回 hidden_dim 并加回投影 state
（残差）→ `Linear(d→d//2) → LayerNorm → Tanh → Linear(d//2→1)` → softmax；
训练记录 Categorical `log_prob`，评估/更新用 greedy argmax。
启用 `--use-sampler-edge-relations` 时，candidate 投影额外融合候选到已选子图的
7 维关系聚合；该开关仅适用于 `--sampler rl`，可与 Predictor 开关独立消融。

超参默认：`--epochs 10 --batch-size 32 --eval-batch-size 64 --hidden-dim 256
--max-steps 10 --k-hops 1 --gnn-layers 2 --heads 4 --dropout 0.1
--sampler-lr 1e-4 --predictor-lr 1e-3 --reinforce-gamma 1.0`；relation-aware
Predictor 与 Sampler 分别通过 `--use-edge-relations` 和
`--use-sampler-edge-relations` 显式开启。
实验推荐配置（沿用至今）：`h128 / γ0.9 / 20ep / ms5`（来源见 §6.3）。

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

### 5.2 数据喂入与计算

- 训练循环用 `torch.randperm` + 批量 fancy-index gather 直接从预载 tensor 取数，
  不经过 DataLoader（内存驻留、无 I/O 负载下快 1–3 个数量级）。
- 全图邻接（不可变 `tuple[frozenset]`）**一次性构建**，训练 batch、每 epoch
  验证、最终测试共享；目标边由 `_TargetSafeAdjacency` 惰性视图逐 target 排除。
- RL 训练耗时主体是 step-graph 前向与 sampler 更新（static/random/heuristic
  模式无此开销，速度快一个量级）；attention 读出开销与图规模成正比。

## 6. 实验记录（结果总表 + 主要结论）

除注明外均为 SHS27k bfs、全图 KG、`h128/g0.9/20ep/ms5` 协议、seed 42/111/123、
CPU。指标为 best-checkpoint（by valMacAUC）test 均值±std；上下文规模为 train
目标实测的每图节点数。

### 6.1 mean-pool 读出（bfs，3 线程）

| 条件 | 上下文(节点) | MacAUC | MacF1 | 秒 |
|---|---:|---|---|---:|
| RL ms0（G0-only） | 2 | 0.7497±0.0120 | 0.5045±0.0183 | 87 |
| rnd 6..7（同 RL 预算随机） | ~6.5 | 0.7586±0.0048 | 0.5146±0.0219 | 101 |
| RL ms5 | ~7 | 0.7758±0.0196 | 0.5515±0.0555 | 1136 |
| RL ms10 | ~10 | 0.7668±0.0109 | 0.5425±0.0160 | 2950 |
| RL ms20 | ~20 | 0.7785±0.0163 | 0.5484±0.0621 | 6664 |
| rnd 19..21 | ~20 | 0.7829±0.0027 | 0.5709±0.0129 | 203 |
| heur 6..7（共邻优先） | ~6.5 | 0.7841±0.0038 | 0.5726±0.0187 | 114 |
| heur 19..21 | ~20 | 0.7840±0.0058 | 0.5534±0.0254 | 183 |
| P2a：RL+结构特征+启发式先验 | ~7 | 0.7716±0.0099 | 0.5438±0.0315 | 1600 |
| R1：RL+margin 奖励 | ~7 | 0.7794±0.0058 | 0.5443±0.0154 | 1427 |
| RL k2 ms5 | ~7 | 0.7716±0.0116 | 0.5587±0.0119 | 1690 |
| rnd k2 6..7 | ~6.5 | 0.7589±0.0134 | 0.5421±0.0209 | 108 |
| **static k1（G0+全 1-hop）** | ~45.6 | **0.8047±0.0046** | **0.5947±0.0123** | 337 |
| static k2（全 2-hop） | ~334 | 0.7899±0.0034 | 0.5695±0.0088 | 2928 |

注：RL ms5 另有同环境复跑 0.7787±0.0086 / 0.5532±0.0434（1497s）；区域规模
k1 mean/median/max = 45.6/41/156 节点（238/131/1451 边），k2 = 334.5/310/835
（2147/1854/5294 边）。

### 6.2 attention 读出与 A2（bfs）

| 条件 | 上下文(节点) | MacAUC | MacF1 | 秒 | 线程 |
|---|---:|---|---|---:|---|
| V2：RL ms5+margin+attention | ~7 | 0.7757±0.0098 | 0.5343±0.0394 | 1364 | 3 |
| V3：heur 6..7+attention | ~6.5 | 0.7836±0.0056 | 0.5460±0.0094 | 262 | 3 |
| **V1a：static k1+attention** | ~45.6 | **0.8105±0.0114** | **0.6158±0.0156** | 599 | 3 |
| V1b：static k2+attention | ~334 | 0.8065±0.0053 | 0.5767±0.0219 | 4038 | 3 |
| E1：A2 底座∪RL 增补+margin(base) | 底座+5 | 0.8064±0.0043 | 0.5868±0.0150 | 4924 | 2 |
| E2：A2+uniform 增补 | 同上 | 0.8008±0.0100 | 0.5821±0.0097 | 3743 | 2 |
| V1a-rel：V1a+train 边 relation（GAT edge_dim=7） | ~45.6 | 0.7911±0.0116 | 0.5556±0.0192 | 691 | 2 |

注：V1a-rel 对同环境 20ep 锚（§6.3 对照）配对：MacAUC −0.0125±0.0172、
MacF1 −0.0291±0.0371（均 1/3 正，bestEp 9/13/4）→ relation 注入在 static k1
下无增益（耗时仅 +2%）。

### 6.3 跨 split、训练时长与历史时代

**V1a 跨 split（2 线程）**：dfs **0.8766±0.0040 / 0.6811±0.0153**（876s，
BS 0/ES 1215/NS 314）；random **0.9658±0.0003 / 0.8313±0.0114**（945s，
BS 1377/ES 138/NS 10，两端均训练可见，仅健全性记录，不与 bfs/dfs 比较）。

**V1a 训练时长（bfs，2 线程）**：40ep 0.8043±0.0086 / 0.5900±0.0272（1345s，
bestEp 11/38/11）vs 20ep 对照 0.8037±0.0075 / 0.5847±0.0188（679s，
bestEp 11/13/11）——同环境配对 diff MacAUC/MacF1 均仅 1/3 正（s42/s123 精确为
0：ep21–40 从未超过 ep11，选中同一 checkpoint）→ **20ep 饱和**。

**dfs 历史套件（mean-pool，3 线程）**：static 0.8676/0.6714 > ms5
0.8493/0.6354 > rnd 0.8387/0.6273 ≈ ms0 0.8371/0.6174（bfs 可见性
BS 0/ES 1078/NS 460，dfs 0/1215/314——NS 占比 20.5% vs 29.9%）。

**split-local 旧时代（已失效，仅背景）**：旧最优 `res-score + h128/γ0.9/20ep/
ms5`（bfs MacF1 0.5239）；当时 dfs 上 ms0 最优——全图 KG 下反转（两个划分均为
ms5 > ms0）；SHS148k random 指标虚高（test BS 96.5%，退化为模式匹配）；40ep
曾为成本-收益甜点。

### 6.4 主要结论（决策链）

1. **上下文量主导**：static k1（全 1-hop，~46 节点）是全部条件最优；同预算下
   RL≈随机（7 节点：rnd−ms5 方向不齐 1/3；20 节点：rl−rnd 五项全负）、6–7 节点
   上下文 ≈ G0-only；上下文量非单调——static k2（334 节点）反而低于 static k1
   （MacAUC −0.015、MacF1 −0.025）→ 最优区间在 k1。
2. **RL 选取假设关闭（四条修复路径全部无效）**：P2a 结构特征+先验初始化
   （0.7716，配对低于 heur7，先验第 1 步保留但 2–5 步漂移 → 特征空间不是瓶颈，
   奖励信号是）；R1 margin 奖励（与 R0 持平但 valMacAUC 方差降 4×，margin
   0→0.26 单调可学——训练目标可优化 ≠ 测试提升）；attention 读出下 RL 从零
   （V2）与 heur（V3）无增益；A2 底座∪增补 E1≈E2（+0.006 MacAUC，2/3，随机
   增补不可区分）且任何增补劣于纯底座（对 V1a 的 MacF1 0/3，−0.029/−0.034）。
   底座播种把 RL 从 0.776 抬到 0.806（E1−V2 = +0.031，3/3），但增益来自底座
   而非选取。唯一例外：k2 大候选空间下 rl−rnd 首次全正（4 项 3/3）——候选
   空间大时选取有区分度，但不改变 static k1 全局最优。
3. **读出与边特征编码**：上下文充足时 target 锚定 attention 显著优于 mean-pool——V1a 对
   static mean-pool bfs MacAUC +0.006（2/3）、MacF1 +0.021（3/3），dfs
   MacAUC +0.0089（3/3）；对 RL ms5 dfs 全面领先（+0.027/+0.046，均 3/3）。
   小预算（~7 节点）0.78 平台不动（V2≈R1、V3 MacF1 微降）——平台是小上下文
   的属性；k2 稀释未被注意力根除（attention k2−k1：MacF1 −0.039）。边类型
   注入 GAT（V1a-rel）同样无增益（配对 MacAUC −0.0125、MacF1 −0.0291，均
   1/3 正）——k1 上下文的瓶颈不在边类型编码。
4. **heur 是行为基准**：零参数"共邻优先"规则 7 节点超随机（+0.026 MacAUC、
   +0.058 MacF1，五项 3/3）且 ≥ RL ms5；20 节点下优势消失（heur−rnd ≈ 0）。
   任何 sampler 改动先同预算配对 heur7。
5. **最优配置跨 split 成立**："static k1 + attention（上下文量 + 读出）"在
   dfs 上 0.8766/0.6811，方向与 bfs 一致；dfs 系统性更易（可见性高），
   跨划分不可混比。
6. **训练时长**：20ep 饱和（40ep 无增益，见 §6.3）；"bestEp 顶界"不等于
   "未饱和"——val 噪声 ±0.01 下顶界只是噪声与截断的巧合。
7. **运维纪律（线程数）**：同 seed 下 OMP/MKL 线程数 2 vs 3 使 CPU 浮点归约
   顺序不同、轨迹从早期分叉，指标漂移达 seed 量级（MacAUC −0.0069±0.0076、
   **MacF1 −0.0311±0.0030，0/3**）→ 配对实验必须固定线程数，跨线程环境的
   历史比较（如 §6.2 的 2/3 线程混排）不可作因果解释。

### 6.5 实验纪律

- 指标方差大（MacF1 seed 间 std 可达 0.05+），结论必须 3-seed（42/111/123）
  配对均值±std 并报正向 seed 数；
- 默认模式推理特征不含任何标签信息（relation-aware 仅显式开启时读 train 边
  标签，val/test 与目标边从不进入，见 §2）；
- 报告结果须标注数据集、split 与线程环境；
- RL 实验默认带 `--reward-margin`（margin 学得动、训练更稳、诊断免费）。

## 7. 验证

```bash
python -m unittest discover -s tests -v    # 101 项
python -m compileall -q src tests
git diff --check
```
