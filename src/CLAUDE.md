# 核心代码设计

系统由 `PPIGraph`、`SubgraphSampler`、`PPIPredictor` 和 `AlternatingTrainer` 组成。输入为目标蛋白对及 2560 维 ESM-2 嵌入，输出为 7 维 PPI 多标签 logits。

## PPIGraph

- `PPIGraph(name, split, root, device, cache_dir)` 加载一个数据集和一种 split。
- actions 文件中的同一无向蛋白对按 `mode` 做 OR 聚合，得到 7 维 multi-hot 标签。
- `get_ppi_indices(split_name)` 返回 split 内的 PPI index。
- `build_graph(split_name="train", undirected=True)` 按当前 split 构建图；默认补齐反向边。
- 构图结果包含：
  - `edge_index`：局部节点索引；
  - `node_index`：local → global 蛋白索引；
  - `node_feat`：当前 split 节点的 embedding。
- `build_graph` 不再返回 `edge_label`；训练使用的每对 PPI multi-hot 标签直接由
  `train_shs27k` 从 `ppi_labels` 读取。
- split 节点始终局部化，不再提供或支持 `remap_nodes` 参数。
- `get_dataloader()` 是公共 DataLoader 接口；当前训练入口为了每个 epoch 使用 `torch.randperm`，手动按 batch 更新。
- `train_shs27k` 通过 `--split {bfs,dfs,random}` 选择 SHS27k 划分，默认是 `bfs`。

## SubgraphSampler

### 输入与防泄漏

- `sample(node_features, edge_index, target_nodes, node_index=None, training=None, adjacency=None)` 的拓扑和特征必须来自同一个 split。
- `node_index` 将目标的 global protein index 映射到当前图的局部行号。
- 每个目标采样前排除目标边的两个方向；目标边不会进入 baseline、initial 或 step graph。
- 邻接表按 split 构建一次并共享（不可变 `tuple[frozenset]`），目标边在采样时惰性排除，避免每 target 深拷贝 O(E) 邻接；目标局部行号用 `torch.searchsorted` 在有序 `node_index` 上二分定位，替代每 target O(N) dict 查找。
- `baseline_graph` 只包含两个目标节点且无边。
- 删除目标边后，安全邻接为空的目标被视为孤立节点。
- 孤立目标从当前 split 的非目标节点中按余弦相似度选择代理；代理不能是目标节点，两个目标可以共享代理。
- 代理加入后，补齐 initial graph 中已选节点之间的全部真实安全边。虚拟代理边和真实边都进入 Predictor，当前不区分 edge type。

### 轨迹与动作

- 状态是当前已选节点 embedding 的均值。
- 当前子图 state 是已选节点 ESM embedding 的均值。state 使用独立的 `state_proj: Linear(esm_dim→hidden_dim)`，每个候选邻居使用独立的 `neighbor_proj: Linear(esm_dim→hidden_dim)`，分别得到 `state_repr` 和 `neighbor_repr`。
- 对每个候选构造 `[state_repr || neighbor_repr]`，输入两层 pairwise MLP：`Linear(2*hidden_dim→action_hidden) → LeakyReLU(0.2) → Linear(action_hidden→1)`，得到动作 score，再在候选集合内 softmax。第一层作用于完整拼接向量，因此可混合 state 与 candidate 特征；不能将 LeakyReLU 先逐元素作用于拼接向量后再与单个向量点积，否则 state 项会在候选 softmax 中抵消。
- pairwise MLP 的 Linear 权重使用 Xavier uniform 初始化、偏置初始化为 0；`state_proj`、`neighbor_proj` 和 action MLP 都参与 `log_prob` 的反向传播。该结构替换了旧的 Query/Key 缩放点积及独立投影线性打分实现。
- 训练时从 `Categorical(softmax(scores))` 随机采样并记录 `log_prob`；评估时用 `argmax`，其 `log_prob` 为零标量。
- Value 由独立的两层 MLP 从状态估计。
- `initial_graph` 的目标节点和代理作为 k-hop 区域种子；区域在删除目标边后的安全邻接上计算，默认 `k_hops=3`。
- 每一步候选为 `frontier(selected, adjacency) ∩ allowed_nodes`，其中 `allowed_nodes` 是上述安全图 k-hop 区域。
- 选中节点后加入它与当前已选节点之间的全部真实安全边，再生成一个 `SamplingStep`。
- 轨迹在 `max_steps` 次动作后或 frontier 为空时结束；没有 STOP 动作。
- 当前没有双目标平衡约束，因此 k-hop 限制不能保证两端都扩展或最终图连通。

### Pairwise MLP BFS 实验

- 配置：SHS27k BFS，`hidden_dim=512`、`max_steps=10`、`k_hops=3`、
  `reinforce_gamma=0.95`、3 层 GAT、seed 42、10 Epoch。
- 验证 Macro-AUC 最佳为 Epoch 9 的 `0.7395`；对应测试
  Macro/Micro-AUC 为 `0.7347/0.7607`，Macro/Micro-F1 为
  `0.5192/0.5629`。
- 测试 ES（1078）Macro/Micro-AUC 为 `0.7471/0.7757`，Macro/Micro-F1 为
  `0.5455/0.5853`；NS（460）为 `0.7045/0.7224`、`0.4542/0.5078`；
  BS 为 0。第 10 Epoch 验证 Macro-AUC 回落到 `0.6898`。
- 相同配置的 seed 123 实验在 Epoch 10 达到最佳验证 Macro-AUC `0.7347`；
  对应测试 Macro/Micro-AUC 为 `0.7216/0.7452`，Macro/Micro-F1 为
  `0.4732/0.5184`。两次实验的完整汇总见
  [`experiments/SHS27K_BFS_pairwise_MLP.md`](../experiments/SHS27K_BFS_pairwise_MLP.md)。

## PPIPredictor

- 输入节点特征先经过 `Linear → LayerNorm → ReLU → Dropout`。
- 图编码使用多层 `GATConv → ELU → Dropout → residual → LayerNorm`；`PPIPredictor`
  模块默认 hidden 512、3 层 GAT，而训练入口 CLI 默认是 hidden 256、2 层
  （见 `train_shs27k.py`），复现实验必须显式传入 `--hidden-dim`/`--gnn-layers`。
- readout 拼接：
  - `h_u + h_v`；
  - `|h_u - h_v|`；
  - 子图节点表示的均值。
- 输出 7 维 logits；推理时对 logits 做 sigmoid 得到概率（训练入口直接在
  `evaluate` 中使用 `torch.sigmoid`）。
- `forward()` 支持多图 batch，通过 `batch` 向量区分子图。
- 全图均值会使距离目标超过 GAT 层数的节点仍参与预测。

## AlternatingTrainer

### Sampler 更新

1. 冻结 Predictor，Sampler 用随机动作生成每个目标的 trajectory。
2. Predictor 批量计算 baseline graph 和每个 step graph 的 BCE loss。
3. 对每一步计算 `r_t = baseline_loss - step_loss`。
4. 对每条 trajectory 从后向前计算 `G_t = r_t + gamma * G_{t+1}`。
5. 使用：

   ```text
   policy_loss = mean(-log_prob_t * stop_gradient(G_t - value_t))
   value_loss  = mean(MSE(value_t, G_t))
   sampler_loss = policy_loss + reinforce_baseline_coef * value_loss
   ```

6. 仅 Sampler 优化器更新参数。

`log_prob` 对策略投影参数可导；离散节点索引、set 邻接和 k-hop mask 是环境/动作空间逻辑，不通过普通反向传播求导。受限候选集合内的 softmax 仍是实际采样策略。

### Predictor 更新

1. 冻结 Sampler，使用贪心动作重新生成 trajectory。
2. 每条 trajectory 只取 `final_graph`；若没有动作，则取含代理的 `initial_graph`。
3. 合并多个变长子图的 feature、offset edge index、target index 和 batch 向量。
4. GAT 输出 logits，使用 BCE with logits 更新 Predictor。

因此 Predictor 训练目标与评估目标一致：都是每个 PPI 的最终子图。

## 评估

- `evaluate()` 在 val/test split 上使用 greedy Sampler 和每条 trajectory 的 `final_graph`。
- logits 经 sigmoid 得到概率；报告 Macro/Micro ROC-AUC 和固定 0.5 阈值的 Macro/Micro F1。该阈值遵循同类论文的评估约定，不在验证集或测试集上调节。
- 额外按测试 PPI 两端相对训练节点集合的可见性分组：
  - BS：两端都在训练节点集合；
  - ES：恰好一端在训练节点集合；
  - NS：两端都不在训练节点集合。
- 空分组不计算指标，返回 `count=0` 和 `None` 指标。

## 当前不变量

- 采样子图不包含目标边的任一方向。
- 采样节点、proxy 和扩展边都属于当前 split。
- 虚拟代理不会是目标节点；两个目标可以共享代理。
- initial graph 会保留已选节点之间的真实安全边。
- Predictor 的目标对表示对端点交换不变。
- RTG 按单条 trajectory 从后向前计算，不跨 trajectory 串联。
