# 核心代码设计

## PPIGraph

- `PPIGraph(name, split, root, device, cache_dir)` 加载一个数据集及其 split。
- `build_graph(split_name="train", undirected=True)` 构建 split-local 图，并保留公共 `edge_label` 字段。
- `edge_index` 是局部节点索引；`node_index` 是 local → global 映射；`node_feat` 是节点 embedding。
- 支持组合：SHS27k 的 `bfs/dfs/random`，SHS148k 的 `dfs/random`，STRING 的 `dfs`。
- 不提供 `remap_nodes` 选项；split-local 是固定语义。

## SubgraphSampler

### 输入与安全性

- `sample(node_features, edge_index, target_nodes, node_index=None, training=None, adjacency=None)` 的特征、节点和边必须来自同一 split。
- `node_index` 必须严格递增，用于二分定位 global target 到 local 行号。
- 目标边的两个方向在每个 target 采样前从安全邻接中排除。
- 代理只能来自当前 split 的非目标节点；两个目标可以共享代理。

### G0 与轨迹

- `baseline_graph` 是唯一初始图 `G_0`。
- 若目标安全邻接为空，按 embedding 余弦相似度选择 proxy。
- `G_0` 只包含目标节点和必要的虚拟 proxy，不预先采样安全一跳邻居。
- 从 `G_0` 的 target/proxy 种子沿安全 adjacency 构造一次 `k_hops` 区域，默认 `k_hops=1`；安全一跳邻居由初始 frontier 提供给后续 action。
- 后续候选为该区域内的当前 frontier；`max_steps` 只限制动作次数，默认值为 `10`，无 STOP 动作。
- `SamplingTrajectory.final_graph` 返回最后一步图；若没有动作则返回 `baseline_graph`。

### Action score

状态是当前已选节点 embedding 的均值。state 和 candidate 使用不同的线性投影，拼接后输入 pairwise MLP：

```text
Linear(2*hidden_dim → action_hidden)
→ LeakyReLU(0.2)
→ Linear(action_hidden → 1)
→ candidate softmax
```

训练时记录 Categorical action 的 `log_prob`；评估时使用 greedy argmax。Sampler 参数通过 REINFORCE 更新，离散节点选择、Python 邻接和 frontier 属于环境逻辑，不通过普通反向传播求导。

## AlternatingTrainer

Sampler 更新阶段：

1. 冻结 Predictor，随机生成 trajectory。
2. 计算 `G_0` 和所有 step graph 的 Predictor BCE loss。
3. 使用增量奖励 `r_t = L_{t-1} - L_t`；第一步以 `G_0` loss 为前项，不包含 `Δn` 或其他复杂度惩罚。
4. 对每条 trajectory 从后向前计算 `G_t = r_t + gamma * G_{t+1}`。
5. 汇总同一 batch 的 detached `G_t - V(s_t)`，按 batch 均值和总体标准差标准化后计算 `policy_loss`；`value_loss` 仍回归原始 `G_t`，再更新 Sampler。

Predictor 更新阶段：

1. 冻结 Sampler，使用 greedy trajectory。
2. 只使用 `final_graph` 训练 Predictor；无动作时使用 `G_0`。
3. 使用 BCE with logits 更新 Predictor。

## PPIPredictor

- 输入特征经过线性层、LayerNorm、ReLU 和 Dropout。
- 图编码使用多层 GAT，并通过 residual/LayerNorm 稳定训练。
- readout 使用 `h_u+h_v`、`|h_u-h_v|` 和子图节点均值。
- 输出 7 维 logits；公共 `predict_proba()` 保留并执行 sigmoid。
- 支持多图 batch。

## 评估

- 训练期间仅评估验证集。
- 训练结束后在验证 Macro-AUC 最佳状态上评估一次测试集。
- 报告 Macro/Micro ROC-AUC 和固定 0.5 阈值的 Macro/Micro F1。
- 测试集按两端节点相对训练节点集合的可见性报告 BS、ES、NS。
