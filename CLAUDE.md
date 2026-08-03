# Protein-Protein Interaction 项目设计

## 子图采样器 Sampler

输入为 PPI 对 (u, v)，初始子图 `G_0 = {u, v}`，逐步扩展子图规模，每步选择当前子图相邻节点（不在当前子图内）中的一个，要求选择的过程是可学习的，直至达到扩展上限参数 `T_max` 或当前子图无相邻节点。

具体来说：

- 每一个时间步 `t = 0, 1, ...`，对应一个子图 `G_t`
- 状态 `s_t` 为当前子图所有节点 esm embedding 的 **均值**（`pooling_mode="mean"`；原始设计为 sum，但 sum 会随子图增大导致 scale 漂移，不推荐）
- 对子图的相邻节点 `i`，结合它的 esm embedding 和与 `G_t` 连边的关系（multi-hot 边特征经 max-pool 聚合），通过 MLP 形成表征 `neighbor_i`
- `s_t` 和相邻节点的表征 `neighbor_i` 通过 **Scaled Dot-Product Attention** 计算相似性：`score_i = dot(Query(state), Key(neighbor_i)) / sqrt(d)`，softmax 后输出概率分布
- 包含一个 **Baseline 价值网络** `V(s_t)`，用于 REINFORCE 方差缩减
- 训练时从 Categorical(probs) 采样节点，记录 log_prob，通过 REINFORCE 策略梯度更新；测试时 argmax 选择概率最大的节点
- 见 [model/sampler.py](model/sampler.py) 和 [model/REPORT.md](model/REPORT.md) 了解完整细节

## 预测器 Predictor

- 子图 `G_{t+1}` 输入 GNN（GAT + 残差连接 + LayerNorm），经 pairwise readout（`[h_u; h_v; h_u⊙h_v; |h_u−h_v|]`）后通过 MLP 输出 **7 维 sigmoid 概率**（多标签分类，非 softmax 单标签），与 label（multi-hot）计算 BCE loss
- 训练时，轮流固定 Sampler 和 Predictor，交替训练：
  - **Sampler 步**：固定 Predictor，REINFORCE 更新（reward = -BCE）
  - **Predictor 步**：固定 Sampler（argmax 模式），BCE 监督更新
- 见 [model/predictor.py](model/predictor.py) 和 [model/ppi_model.py](model/ppi_model.py) 了解完整细节

## 已知设计注意事项

1. **状态池化**：原始设计为 sum，改为 mean（默认）以避免 scale 漂移。`pooling_mode="sum"` 可切回原设计。
2. **边特征泄露**：Sampler 使用 ground truth 边标签作为邻居特征，存在信息泄露风险。设 `use_edge_features_in_sampler=False` 可退化为二值连接指示器。
3. **REINFORCE 高方差**：离散采样 + 稀疏奖励。Baseline 网络可缓解，后续可考虑 Gumbel-Softmax 替代。
4. **无法批量化**：各 PPI 对子图扩展路径不同。通过 `grad_accum_steps` 梯度累积缓解。
5. **孤立 PPI 对**：部分蛋白质对无邻居（如 ppi=1994），已处理 frontier 为空的退化情况。
