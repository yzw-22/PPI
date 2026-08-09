# 核心代码设计

系统由 `PPIGraph`、`SubgraphSampler`、`PPIPredictor` 和 `AlternatingTrainer` 组成。输入为目标蛋白对及 2560 维 ESM-2 嵌入，输出为 7 维 PPI 多标签 logits。

## PPIGraph

- `PPIGraph(name, split, root, device, cache_dir)` 加载一个数据集和一种 split。
- actions 文件中的同一无向蛋白对按 `mode` 做 OR 聚合，得到 7 维 multi-hot 标签。
- `get_ppi_indices(split_name)` 返回 split 内的 PPI index。
- `build_graph(split_name="train", undirected=True)` 按当前 split 构建图；默认补齐反向边。
- 构图结果包含：
  - `edge_index`：局部节点索引；
  - `edge_label`：边标签；
  - `node_index`：local → global 蛋白索引；
  - `node_feat`：当前 split 节点的 embedding。
- split 节点始终局部化，不再提供或支持 `remap_nodes` 参数。
- `get_dataloader()` 是公共 DataLoader 接口；当前训练入口为了每个 epoch 使用 `torch.randperm`，手动按 batch 更新。

## SubgraphSampler

### 输入与防泄漏

- `sample(node_features, edge_index, target_nodes, node_index=None, training=None, adjacency=None)` 的拓扑和特征必须来自同一个 split。
- `node_index` 将目标的 global protein index 映射到当前图的局部行号。
- 每个目标采样前删除目标边的两个方向；目标边不会进入 baseline、initial 或 step graph。
- `baseline_graph` 只包含两个目标节点且无边。
- 删除目标边后，安全邻接为空的目标被视为孤立节点。
- 孤立目标从当前 split 的非目标节点中按余弦相似度选择代理；代理不能是目标节点，两个目标可以共享代理。
- 代理加入后，补齐 initial graph 中已选节点之间的全部真实安全边。虚拟代理边和真实边都进入 Predictor，当前不区分 edge type。

### 轨迹与动作

- 状态是当前已选节点 embedding 的均值。
- Query、Key 线性投影得到动作 score，使用缩放点积。
- 训练时从 `Categorical(softmax(scores))` 随机采样并记录 `log_prob`；评估时用 `argmax`，其 `log_prob` 为零标量。
- Value 由独立的两层 MLP 从状态估计。
- `initial_graph` 的目标节点和代理作为 k-hop 区域种子；区域在删除目标边后的安全邻接上计算，默认 `k_hops=3`。
- 每一步候选为 `frontier(selected, adjacency) ∩ allowed_nodes`，其中 `allowed_nodes` 是上述安全图 k-hop 区域。
- 选中节点后加入它与当前已选节点之间的全部真实安全边，再生成一个 `SamplingStep`。
- 轨迹在 `max_steps` 次动作后或 frontier 为空时结束；没有 STOP 动作。
- 当前没有双目标平衡约束，因此 k-hop 限制不能保证两端都扩展或最终图连通。

## PPIPredictor

- 输入节点特征先经过 `Linear → LayerNorm → ReLU → Dropout`。
- 图编码使用多层 `GATConv → ELU → Dropout → residual → LayerNorm`；默认模块配置为 hidden 512、3 层 GAT，但训练 CLI 默认参数可覆盖它。
- readout 拼接：
  - `h_u + h_v`；
  - `|h_u - h_v|`；
  - 子图节点表示的均值。
- 输出 7 维 logits；`predict_proba()` 使用 sigmoid。
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
- logits 经 sigmoid 得到概率；报告 Macro/Micro ROC-AUC 和固定 0.5 阈值的 Macro/Micro F1。
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
