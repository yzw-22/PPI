# 核心代码设计

系统由 `PPIGraph`、`SubgraphSampler`、`PPIPredictor` 和 `AlternatingTrainer` 组成。输入为目标蛋白对及 2560 维 ESM-2 嵌入，输出为 7 维 PPI 多标签 logits。

## PPIGraph

- 从 actions 文件按无向蛋白对 OR 聚合标签。
- `build_graph()` 按 train/val/test 单独建图，默认补全反向边。
- 当前训练使用 `remap_nodes=False`，使 `edge_index`、`node_index` 和特征行采用全局蛋白索引。
- `get_dataloader()` 是可用的公共接口；当前训练入口使用 `torch.randperm` 手动切 batch。

## SubgraphSampler

### 初始化与防泄漏

- 输入拓扑必须来自当前 split。
- 每个目标对采样前移除目标边的两个方向。
- `baseline_graph` 只含 `u、v`，不含边。
- 移除目标边后，安全邻接为空的目标节点被视为孤立节点。
- 孤立目标从目标节点以外的全部节点中选择余弦相似度最高者作为虚拟代理；两个目标可以共享代理，节点列表保持唯一。
- 虚拟代理加入后，补齐已选节点之间的真实安全边。虚拟边与真实边均进入采样子图。

### 轨迹

- 状态是当前已选节点 ESM 嵌入的均值。
- 动作空间是与当前子图相邻但尚未选择的节点。
- Query 和 Key 分别由线性投影得到，动作分数使用缩放点积。
- 训练时从 `Categorical` 采样并记录 `log_prob`；评估时使用 `argmax`。
- Value 由独立两层 MLP 从状态估计。
- 每次动作加入一个节点及其与已选节点之间的全部真实边；达到 `max_steps` 或前沿为空时结束。
- `max_steps` 是动作数上限，不是 hop 深度上限，当前没有 STOP 动作。

## PPIPredictor

- 节点编码：`Linear + LayerNorm + ReLU + Dropout`。
- 图编码：多层 `GATConv + ELU + Dropout + 残差 + LayerNorm`。
- Readout：拼接 `h_u+h_v`、`|h_u-h_v|` 和子图节点均值。
- 输出 7 维 logits；`predict_proba()` 对单图 logits 应用 sigmoid。
- 全图均值直接参与输出，因此距离目标超过 GAT 层数的节点仍会影响预测。

## AlternatingTrainer

训练接口仅保留 batch 版本，但每条可变长度轨迹仍逐目标生成。

### Sampler 更新

1. 冻结 Predictor，Sampler 使用随机动作生成轨迹。
2. Predictor 批量计算无边基线图和所有 step 子图的 BCE loss。
3. 对每一步计算 `reward = baseline_loss - step_loss`。
4. 使用 `policy_loss = -log_prob × stop_gradient(reward-value)` 和 `value_loss = MSE(value,reward)` 更新 Sampler。

所有 step 等权平均；轨迹较长的样本贡献更多项，reward 不按节点数或轨迹长度归一化。

### Predictor 更新

1. 冻结 Sampler，以贪心动作重新生成轨迹。
2. 使用每条轨迹的全部 step 子图；无 step 时使用基线图。
3. 拼接节点、偏移边索引并构造 batch 向量，一次执行 GAT。
4. 使用 BCE with logits 更新 Predictor。

每个 batch 固定执行“Sampler 更新 → Predictor 更新”。

## 评估

- 分别使用测试 split 的特征和拓扑，Sampler 采用贪心动作。
- 每条轨迹使用 `final_graph`；无 step 时为含代理的 `initial_graph`。
- logits 经 sigmoid 后，以 0.5 为 F1 阈值，报告 Macro/Micro ROC-AUC 和 F1。

## 已验证不变量

- 采样子图不包含目标边的任一方向。
- 虚拟代理不会选择目标节点，可以被两个目标共享。
- 代理初始化后保留已选节点之间的真实安全边。
- Predictor 的目标对表示对端点交换保持不变。
