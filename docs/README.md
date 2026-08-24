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

## 7. 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```
