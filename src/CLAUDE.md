# 任务目标

实现一个基于强化学习的蛋白质-蛋白质相互作用（PPI）预测系统。系统包含两个核心模块：
**子图采样器（SubgraphSampler）** 与 **预测器（PPIPredictor）**，通过**交替训练**
联合优化。本文档描述当前项目的实际架构、训练流程与关键不变量，供后续开发与修改参考。

---

## 1. 总体架构

- **输入**：PPI 节点对 `(u, v)`，以及预计算的 ESM-2 3B 蛋白质嵌入（维度 `esm_dim = 2560`）。
- 采样器从初始子图 `G_0 = {u, v}` 出发，逐步选择邻居节点扩展，至多 `T_max` 步。
- 预测器对子图运行 GAT 消息传递，内部输出 7 维多标签 logits；推理接口再施加 sigmoid。
- 训练时，采样器为 batch 中的每个目标对生成轨迹；冻结的预测器批量计算各个
  `G_t` 的损失，奖励 `r_t = l_0 - l_t` 用于 REINFORCE RL 训练。

### 1.1 数据泄露防护（强制不变量）

核心 PPI 对 `(u, v)` 的直连边**正是待预测的标签**。因此它**在任何路径下都不得
出现在子图的边列表中**——对训练集与测试集一致成立：

- 排除 `(u, v)` 之间的无向边；
- 由此基线子图 `G_0 = {u, v}` 不含任何边，`l_0` 成为"仅靠 u/v 自身节点特征、
  无结构信息"的干净基线。

此外，采样器的状态与邻居表征只使用**节点级 ESM 嵌入**，不使用边标签特征，
从源头避免标签信号进入结构表征。

---

## 2. 子图采样器（src/sampler.py）

### 2.1 图定义与初始化

- 图结构：节点代表蛋白质，边由 PPI 网络提供（`PPIGraph`，邻接表 + 边特征）。
- 输入：PPI 节点对 `(u, v)`。
- 初始子图：`G_0 = {u, v}`（若 `u` 或 `v` 是孤立蛋白，则注入虚拟代理，见 §3）。

### 2.2 采样过程

对于时间步 `t >= 0` 的子图 `G_t`：
- **动作空间**：从前沿 `N_t - G_t`（与子图相邻但不在子图内的节点）中选择一个
  节点加入子图，并加入其与子图的无向连边。
- **终止条件**：达到最大扩展次数 `T_max` 或前沿为空。
- **状态表征 `s_t`**：子图 `G_t` 中所有节点 ESM 嵌入的均值。

### 2.3 动作策略

对候选邻居节点 `i`：
1. 邻居表征即它的 ESM 嵌入，记为 `neighbor_i`。
2. 缩放点积注意力得分 `score_i = (W_Q · s_t) · (W_K · neighbor_i) / sqrt(d)`。
3. Softmax 得概率分布 `probs`。
4. 价值网络 `V(s_t)` 估计状态价值，用于 REINFORCE 方差缩减。

### 2.4 训练与推理模式

- **训练模式**：从 `Categorical(probs)` 采样节点，记录 `log_prob`。
- **推理模式**：`argmax` 选择概率最大的节点。

---

## 3. 孤立节点处理机制

### 3.1 定义

在 PPI 对 `(u, v)` 中，若 `u` 仅与 `v` 有边，则 `u` 为孤立节点

### 3.2 虚拟代理注入

对孤立节点 `u` 或 `v`，计算所有节点与它的余弦相似度，取最大值对应的节点作为**虚拟代理**，在代理节点与被代理节点中加入一条虚拟无向边

- 候选集合排除目标节点 `u`、`v`；两个目标节点可以共享同一个代理，但选中节点在子图中只保留一次。
- 代理加入后，初始子图补入所有已选节点之间存在的真实安全边；目标边仍被排除。

### 3.3 决策记录：余弦相似度数值特征

当前实现**仅用余弦相似度选择虚拟代理节点**，未将相似度数值作为额外特征
拼接到状态向量或邻居表征中。保持该策略。

---

## 4. 预测器（src/predictor.py）

### 4.1 架构

- **节点投影**：`Linear(esm_dim → hidden) + LayerNorm + ReLU + Dropout`。
- **图神经网络**：多层 `GATConv`（torch_geometric）+ LayerNorm + 残差连接，
  `gnn_heads` 个注意力头，输出拼接回 `hidden` 维。
- **Readout**：获取整个子图的表征，将其与 u, v 结合，注意 u, v 顺序不应影响拼接后的向量。
- **输出层**：输出 7 维 logits；训练使用 `BCEWithLogitsLoss`，推理时施加 sigmoid。

### 4.2 损失与奖励构造

- **损失**：7 维概率与 multi-hot 标签的 Binary Cross-Entropy（BCE）。
- **基线损失 `l_0`**：初始子图 `G_0 = {u, v}`（无任何边）输入预测器的 BCE，对于有虚拟代理的情况，也是只输入 `u, v`，代表不采用子图提取方法的基线。
- **子图损失 `l_t`**：当前子图 `G_t` 输入预测器的 BCE。
- **奖励**：`r_t = l_0 - l_t`（正值表示引入子图信息带来的性能提升）。

---

## 5. 训练策略：交替训练

### 5.1 Sampler 批量训练步

- **固定** Predictor（`requires_grad = False`），Sampler 为训练模式。
- 对 batch 中每个目标对生成轨迹，随后批量计算基线和各步损失：
  1. 计算基线损失 `l_0`（子图 `{u, v}`）；
  2. 运行 Sampler，得到各目标对的 `steps`；
  3. 用冻结的 Predictor 批量计算 `l_t`，得到 `r_t = l_0 - l_t`；
  4. `advantage = r_t - V(s_t)`；
  5. `policy_loss = -log_prob * advantage`；
  6. `value_loss = MSE(V(s_t), r_t)`；
  7. `loss = policy_loss + β * value_loss`（β = `reinforce_baseline_coef`）。

### 5.2 Predictor 批量训练步

- **固定** Sampler（argmax 推理模式），Predictor 为训练模式。
---

## 6. 评估与推理

测试指标采用 AUC-ROC、MACRO-F1、MICRO-F1。

---

## 7. 当前实现

- `SubgraphSampler`、`PPIPredictor` 和 `AlternatingTrainer` 已实现于 `src/`。
- 采样决策仍按目标对逐样本生成；`AlternatingTrainer` 仅保留批量训练接口，
  将 Predictor/GAT 与损失按 batch 合并。
- `train_shs27k.py` 提供 SHS27k/bfs 的可复现实验入口。
- 采样器接收当前 split 的 `edge_index`，并在构建邻接表前移除目标 PPI 的两个方向。
- 虚拟代理只用于采样轨迹；预测器的基线图始终只有目标节点且无边。
- `PPIPredictor.forward()` 返回 logits；训练使用 `BCEWithLogitsLoss`，推理使用 `predict_proba()` 返回 sigmoid 概率。
- `PPIGraph.build_graph(split_name, remap_nodes=False)` 是当前训练器的推荐输入，保证节点特征索引与全局蛋白索引一致。

### 7.1 已验证不变量

- 采样轨迹中不会出现目标 PPI 的正向或反向边。
- 虚拟代理不会选择目标对端。
- Predictor 的目标对表示使用 `u + v` 与 `abs(u - v)`，交换 `(u, v)` 不改变输出。
- 采样器训练模式记录 `log_prob`，推理模式使用贪心 argmax。
