# 任务目标

实现一个基于强化学习的蛋白质-蛋白质相互作用（PPI）预测系统。系统包含两个核心模块：
**子图采样器（SubgraphSampler）** 与 **预测器（PPIPredictor）**，通过**交替训练**
联合优化。本文档描述当前项目的实际架构、训练流程与关键不变量，供后续开发与修改参考。

---

## 1. 总体架构

- **输入**：PPI 节点对 `(u, v)`，以及预计算的 ESM-2 3B 蛋白质嵌入（维度 `esm_dim = 2560`）。
- 采样器从初始子图 `G_0 = {u, v}` 出发，逐步选择邻居节点扩展，至多 `T_max` 步。
- 预测器对子图运行 GAT 消息传递，经 Pairwise Readout 输出 7 维多标签 Sigmoid 概率。
- 训练时，采样器每扩展一个节点（每个时间步 `t`）都把当前子图 `G_t` 输入冻结的
  预测器，得到该步损失 `l_t`，奖励 `r_t = l_0 - l_t`，用于 REINFORCE 训练。

### 1.1 数据泄露防护（强制不变量）

核心 PPI 对 `(u, v)` 的直连边**正是待预测的标签**。因此它**在任何路径下都不得
出现在子图的边列表中**——对训练集与测试集一致成立：

- 采样器生成子图边时排除 `(min(u, v), max(u, v))`；
- 预测器 `_build_subgraph` 现场重推导边时同样排除该边；
- 由此基线子图 `G_0 = {u, v}` 不含任何边，`l_0` 成为"仅靠 u/v 自身节点特征、
  无结构信息"的干净基线。

此外，采样器的状态与邻居表征只使用**节点级 ESM 嵌入**，不使用边标签特征，
从源头避免标签信号进入结构表征。

---

## 2. 子图采样器（model/sampler.py）

### 2.1 图定义与初始化

- 图结构：节点代表蛋白质，边由 PPI 网络提供（`PPIGraph`，邻接表 + 边特征）。
- 输入：PPI 节点对 `(u, v)`。
- 初始子图：`G_0 = {u, v}`（孤立/松散孤立 PPI 对注入虚拟代理，见 §3）。

### 2.2 采样过程

对于时间步 `t >= 0` 的子图 `G_t`：
- **动作空间**：从前沿 `N_t - G_t`（与子图相邻但不在子图内的节点）中选择一个
  节点加入子图。
- **终止条件**：达到最大扩展次数 `T_max` 或前沿为空。
- **状态表征 `s_t`**：子图 `G_t` 中所有节点 ESM 嵌入的均值。

### 2.3 动作策略

对候选邻居节点 `i`：
1. 邻居表征 `neighbor_i = neighbor_mlp(ESM_i)`（仅 ESM 嵌入）。
2. 缩放点积注意力得分 `score_i = (W_Q · s_t) · (W_K · neighbor_i) / sqrt(d)`。
3. Softmax 得概率分布 `probs`。
4. 价值网络 `V(s_t)` 估计状态价值，用于 REINFORCE 方差缩减。

### 2.4 训练与推理模式

- **训练模式**：从 `Categorical(probs)` 采样节点，记录 `log_prob`。
- **推理模式**：`argmax` 选择概率最大的节点。

### 2.5 轨迹记录与边信息

- 训练模式下，每个时间步记录 `SamplerStep`：
  `subgraph_nodes`（扩展后子图节点）、`subgraph_edges`（子图边）、`log_prob`、`value`。
- 整条轨迹返回 `SamplerTrajectory`：
  `initial_nodes`、`steps`、`final_subgraph`、`final_edges`。
  推理模式不记录 `steps`，但 `final_subgraph`/`final_edges` 仍反映真实扩展结果。
- 边以无向 `(min, max)` 规范化、增量维护（每个新节点加入时收集其与子图内邻居
  的边），并**排除核心对 `(u, v)` 直连边**（§1.1 不变量）。
- 边随轨迹存储，预测器直接消费（见 §4.2），避免现场重推导（更快且结构一致）。

---

## 3. 孤立 PPI 对处理机制

### 3.1 定义

孤立（松散孤立）PPI 对定义参见 `dataset/CLAUDE.md`；代码中实现为
**至少一端度数 ≤ 1**（`deg_u > 1 and deg_v > 1` 才算非松散孤立）。

### 3.2 虚拟代理注入

对孤立 PPI 对 `(u, v)`：
- 计算所有节点与 `u`/`v` 的余弦相似度，取最大值对应的节点作为**虚拟代理**
  加入子图。
- 排除 `u`、`v` 自身及其直接邻居。

### 3.3 决策记录：余弦相似度数值特征

当前实现**仅用余弦相似度选择虚拟代理节点**，未将相似度数值作为额外特征
拼接到状态向量或邻居表征中。保持该策略。

---

## 4. 预测器（model/predictor.py）

### 4.1 架构（已定案）

- **节点投影**：`Linear(esm_dim → hidden) + LayerNorm + ReLU + Dropout`。
- **图神经网络**：多层 `GATConv`（torch_geometric）+ LayerNorm + 残差连接，
  `gnn_heads` 个注意力头，输出拼接回 `hidden` 维。
- **Readout**：Pairwise Readout，拼接目标对 `(u, v)` 的表征：
  `[h_u; h_v; h_u ⊙ h_v; |h_u - h_v|]`。
- **输出层**：MLP（`Linear(hidden*4 → hidden) → ReLU → Dropout →
  Linear(hidden/2) → ReLU → Dropout → Linear(num_labels)`），
  输出 7 维 Sigmoid 概率（多标签 BCE）。

### 4.2 子图边构建

`_build_subgraph` 支持两种边来源，结果一致：
- 传入 `edges`（采样器存储的边）：O(E) 的全局→局部索引映射，更快；
- `edges=None` 时从 `graph.adj_list` 现场重推导（O(Σdeg)）。

两条路径均**排除核心对 `(u, v)` 直连边**（§1.1 不变量）。

### 4.3 损失与奖励构造

- **损失**：7 维概率与 multi-hot 标签的 Binary Cross-Entropy（BCE）。
- **基线损失 `l_0`**：初始子图 `G_0 = {u, v}`（无任何边）输入预测器的 BCE。
- **子图损失 `l_t`**：当前子图 `G_t` 输入预测器的 BCE。
- **奖励**：`r_t = l_0 - l_t`（正值表示引入子图信息带来的性能提升）。

---

## 5. 训练策略：交替训练

固定一方、更新另一方，超参数集中在 `model/config.py`（`PPIConfig`）。

### 5.1 Sampler 训练步（`PPIModel.train_sampler_step`）

- **固定** Predictor（`requires_grad = False`），Sampler 为训练模式。
- 对每条 PPI：
  1. 计算基线损失 `l_0`（子图 `{u, v}`）；
  2. 运行 Sampler，得轨迹 `steps`；
  3. 逐时间步用冻结的 Predictor 计算 `l_t`，得 `r_t = l_0 - l_t`；
  4. `advantage = r_t - V(s_t)`；
  5. `policy_loss = -log_prob * advantage`；
  6. `value_loss = MSE(V(s_t), r_t)`；
  7. `loss = policy_loss + β * value_loss`（β = `reinforce_baseline_coef`）。

### 5.2 Predictor 训练步（`PPIModel.train_predictor_step`）

- **固定** Sampler（argmax 推理模式），Predictor 为训练模式。
- 对每条 PPI：用最终子图（`final_subgraph` + `final_edges`）计算 BCE 损失 `l_t`，
  反向传播更新 Predictor。

### 5.3 关键超参数

`T_max`、`gnn_num_layers`、`gnn_heads`、`gnn_dropout`、`hidden_dim`、
`attention_dim`、`lr_sampler`、`lr_predictor`、`sampler_steps`、`predictor_steps`、
`reinforce_baseline_coef`、`isolated_proxy`、`device`。

---

## 6. 评估与推理

- **评估**（`PPIModel.evaluate`）：micro/macro AUC 与 F1-score；
  `tune_threshold` 在验证集上搜索使 F1 最大的决策阈值。
- **推理**（`predict` / `predict_batch`）：Sampler argmax 扩展 → Predictor 输出。
- **批量评估**：`_predict_matrix` 返回预测概率矩阵与标签矩阵。

---

## 7. 文件组织

- `model/config.py`：`PPIConfig`，集中全部超参数。
- `model/graph_utils.py`：`PPIGraph`（邻接表、边特征、`get_frontier`）、`build_graph`。
- `model/sampler.py`：`SubgraphSampler` + 轨迹数据结构 `SamplerStep`/`SamplerTrajectory`。
- `model/predictor.py`：`PPIPredictor`（GAT + Pairwise Readout + MLP）。
- `model/ppi_model.py`：`PPIModel`，交替训练循环、奖励计算、评估与推理。
- `analysis_probe.py`：诊断探针（量化 Predictor 对子图的敏感性）。
- `train_shs27k.py`：SHS27k 训练脚本。

---

## 8. 开发约定

1. **不得破坏 §1.1 的核心边排除不变量**：任何新增的边构建/重推导路径都必须
   排除核心对 `(u, v)` 直连边。
2. 保持"存储边 == 重推导边"的一致性：改动 `_build_subgraph` 或采样器边收集时，
   应验证两条路径在 eval 模式下输出等价。
3. 保持交替训练结构：Sampler 训练（固定 Predictor，REINFORCE）与
   Predictor 训练（固定 Sampler，BCE）的固定/更新关系不可颠倒。
