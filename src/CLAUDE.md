# 核心代码设计

SHS27k 实验结果与复现命令见上级目录的 `docs/` 报告；当前代码说明以本文件和仓库根目录 `CLAUDE.md` 为准。

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
- 对 `u`、`v` 和实际存在的 proxy，分别从各自安全一跳邻居中最多采样 `fixed_num` 个节点，默认 `fixed_num=1`。
- 一跳采样使用独立 seed `42`，重复节点去重。
- `G_0` 保留已选节点之间的全部安全诱导边和虚拟 proxy 边。
- 后续候选为当前 frontier，不使用 hop 限制；最多执行 `max_steps` 次策略决策，默认值为 `10`。
- `fixed_num` 仅限制每个实际初始 seed 的一跳采样数量，`max_steps` 仅限制策略 action 次数；最终图大小取决于安全邻接、proxy 选择、节点去重和 frontier 候选，不使用派生的聚合上下文节点上限。
- 策略包含显式 STOP 决策；达到 `max_steps` 上限时也会终止。
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

### RandomSubgraphSampler（消融）

- 从当前 split 的安全邻接中确定必要的 proxy，再从 `{u, v, proxy}` 的一跳邻居并集随机抽取最多配置数量的上下文节点。
- 使用私有 seed `42`，同一 target 的图可复现；目标边仍双向移除。
- 只返回无 action 的 `baseline_graph`/`final_graph`，训练时跳过 Sampler 更新，仅更新 Predictor。

### RandomIterativeSubgraphSampler（消融）

- 复用 learned Sampler 的 `G0` 和当前 frontier。
- 每步从排序后的 frontier 中用私有 seed `42` 随机选择一个节点，最多扩展 10 次。
- 无可学习参数，不执行 REINFORCE，仅使用最终图更新 Predictor。

CLI 只暴露五种模式：`learned`、`target_only`、`target_proxy`、`random_1hop10`、`random_iterative10`；旧的随机模式和独立上下文预算参数均不再支持。

## AlternatingTrainer

Sampler 更新阶段：

1. 冻结 Predictor，随机生成 trajectory。
2. 计算 `G_0` 和所有 step graph 的 Predictor BCE loss。
3. 普通扩展动作使用增量奖励 `r_t = L_{t-1} - L_t - lambda * delta_extra_nodes`；显式 STOP/DONE 动作奖励为 `0`，复杂度惩罚只计算本步新增节点。
4. 对每条 trajectory 从后向前计算 `G_t = r_t + gamma * G_{t+1}`。
5. 将该 batch 全部动作 step 的 detached `G_t - V(s_t)` 标准化后计算 `policy_loss`；`value_loss` 仍使用原始 `G_t`，再更新 Sampler。

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
