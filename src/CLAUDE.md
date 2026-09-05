# 核心代码设计

## PPIGraph

- `PPIGraph(name, split, root, device, cache_dir)` 加载一个数据集及其 split。
- `build_graph(split_name="train", undirected=True)` 构建 split-local 图，并保留公共 `edge_label` 字段。
- `build_full_graph(undirected=True)` 构建**全数据集全图**（全部蛋白节点 + 全部 PPI 边），训练入口以它作为训练/验证/测试共用的采样 KG。
- `edge_index` 是局部节点索引；`node_index` 是 local → global 映射；`node_feat` 是节点 embedding。
- 支持组合：SHS27k 的 `bfs/dfs/random`，SHS148k 的 `dfs/random`（bfs 由
  `--root dataset_ppisplit` 提供，默认 `dataset/` 无此文件），STRING 的 `dfs`；
  缺失的 split 文件在参数解析阶段报错。
- 不提供 `remap_nodes` 选项；split 只决定目标 PPI 对与训练节点集合，图本身为全图。

## SubgraphSampler

### 输入与安全性

- `sample(node_features, edge_index, target_nodes, node_index=None, training=None, adjacency=None, edge_relations=None)` 的特征、节点和边必须来自同一张图（训练/验证/测试共用全图）。
- `node_index` 必须严格递增，用于二分定位 global target 到 local 行号。
- 目标边的两个方向在每个 target 采样前从安全邻接中排除。
- 代理只能来自当前图的非目标节点（全图即全部蛋白）；两个目标可以共享代理。
- `EdgeRelationLookup` 只收录 train split 边；采样图的 val/test 拓扑边和虚拟
  proxy 边使用全零 7 维 relation，正反方向 relation 相同。目标边删除后才物化
  `edge_attr`，不会暴露当前目标标签。
- `relation_dim` 启用 RL relation-aware 打分；每个候选到当前已选节点的可见关系
  逐维 max/OR 聚合，经无 bias 投影后加到 candidate 表示。聚合只查询目标安全
  adjacency，因此目标边 relation 不会进入策略。
- `structural_features`（`--sampler-structural-features`，RL 专属）开启候选
  拓扑特征（8 维：共邻指示、触目标数、归一化安全度数、已选连接率、有界目标
  距离、Adamic-Adar），经零初始化投影进 candidate 表示，另加先验初始化的
  可训练线性 skip（初始 greedy ≡ heuristic 排序）；两路径均只用安全邻接，
  目标边不可见。
- `base`（`--sampler-base`，RL 专属）：`none`（默认）或 `static`。`static`
  开启 A2 增补语义——候选区域先按 G0 种子固定一次，再把整个 static 1-hop
  底座（`StaticNeighborhoodSampler` 在 `k_hops=1` 下的图，含 proxy 邻域）
  播种进已选集合并作为 `reference_graph` 随轨迹携带；frontier 从全部已选
  节点（含底座）初始化，动作只来自区域剩余的壳层。`k_hops=1` 时 frontier
  恒空，采样严格退化为 static k1。无动作的增补轨迹以 `prediction_graph`
  （= 底座）参与 Predictor 训练与评测。
- `policy`（`--sampler-policy`，RL 专属）：`learned`（默认，网络打分 +
  REINFORCE）或 `uniform`（对照臂：增补动作在训练与评测时都从同一 frontier
  均匀抽取，`log_prob = −log|frontier|` 无梯度，训练入口不构建 sampler
  optimizer，sampler 更新为 no-op）。

### G0 与轨迹

- `baseline_graph` 是唯一初始图 `G_0`。
- 若目标安全邻接为空，按 embedding 余弦相似度选择 proxy。
- `G_0` 只包含目标节点和必要的虚拟 proxy，不预先采样安全一跳邻居。
- 从 `G_0` 的 target/proxy 种子沿安全 adjacency 构造一次 `k_hops` 区域，默认 `k_hops=1`；安全一跳邻居由初始 frontier 提供给后续 action。
- 后续候选为该区域内的当前 frontier；`max_steps` 只限制动作次数，默认值为 `10`，无 STOP 动作。
- `SamplingTrajectory.final_graph` 返回最后一步图；若没有动作则返回 `baseline_graph`。
  `prediction_graph` 是 Predictor 实际使用的图：末步图 → 无步时的
  `reference_graph`（A2 底座）→ `baseline_graph`。
- `StaticNeighborhoodSampler`（消融用，不可学习，零参数）：`final_graph` =
  `baseline_graph` = G0 种子（u、v、proxy）的全部安全 `k_hops` 区域诱导子图，
  无动作；trainer 的 sampler 更新阶段因此为 no-op，只训练 Predictor。与 RL
  Sampler 共享同一候选空间（`_k_hop_region` + 安全邻接），用于检验 RL 选取
  的作用（static 图是任一 RL 轨迹图的信息上界）。
- `RandomSubsetSampler`（消融用，不可学习，零参数）：从 `k_hops` 区域均匀随机
  取 `min_size`~`max_size`（恒含 u/v）个节点的诱导子图，无动作；与 RL 同规模
  对比，用于分离"上下文量"与"选取策略"（random-subset 模式由
  `--sampler random-subset` 开启，规模由 `--random-subset-min/max-size` 控制）。
- `HeuristicSampler`（诊断用，不可学习，零参数）：同预算下按确定性拓扑规则
  选取——u/v 公共邻居 → 单侧邻接 → 区域其余，层内按安全度数降序、id 升序
  （`--sampler heuristic`，规模与 random-subset 共用同一组开关）；作为 RL 的
  行为基准。

### Action score

状态是当前已选节点 embedding 的均值。state 和 candidate 使用不同的线性投影，
拼接后映射回 hidden_dim 并加回投影 state（残差），再经 LN/Tanh 打分头：

```text
Linear(2*hidden_dim → hidden_dim)（+ state 残差）
→ Linear(hidden_dim → hidden_dim//2)
→ LayerNorm(hidden_dim//2)
→ Tanh
→ Linear(hidden_dim//2 → 1)
→ candidate softmax
```

训练时记录 Categorical action 的 `log_prob`；评估时使用 greedy argmax。Sampler 参数通过 REINFORCE 更新，离散节点选择、Python 邻接和 frontier 属于环境逻辑，不通过普通反向传播求导。`policy="uniform"` 时动作改为 frontier 上的均匀分布（训练与评测一致随机），`log_prob = −log|frontier|`，不进入反向传播。

## AlternatingTrainer

Sampler 更新阶段：

1. 冻结 Predictor，随机生成 trajectory。
2. 计算 `G_0` 和所有 step graph 的 Predictor BCE loss。
3. 使用增量奖励 `r_t = L_{t-1} - L_t`；第一步以 `G_0` loss 为前项，不包含 `Δn` 或其他复杂度惩罚。`--reward-margin` 开启替代方案：`r_t` 改为固定参考图的标签对齐平均概率边际改进（`M(p)=mean_j((2y_j−1)·p_j) ∈ [−1,1]`，`--reward-pos/neg` 非对称缩放，默认 2:1）。参照图由 `--reward-ref` 选择：`initial`（默认，G0，历史行为逐位一致）或 `base`（轨迹的 `reference_graph`，即 A2 静态底座预测；bce_diff 的首步锚点与 `mean_final_margin` 的无动作回退同用该参照）。`mean_final_margin` 指标两种模式均输出并进入 epoch 记录。
4. 对每条 trajectory 从后向前计算 `G_t = r_t + gamma * G_{t+1}`。
5. 汇总同一 batch 的 detached `G_t`（无学习 baseline），按 batch 均值和总体标准差标准化后计算 `policy_loss`，再更新 Sampler。`sampler_optimizer is None`（不可学习 sampler 或 uniform 策略）时整个 sampler 更新短路为零指标，`sampler_step_count` 仍报告真实步数。

Predictor 更新阶段：

1. 冻结 Sampler，使用 greedy trajectory。
2. 只使用 `prediction_graph` 训练 Predictor（有动作 → 末步图；无动作 →
   A2 底座 `reference_graph`，否则 `G_0`）。
3. 使用 BCE with logits 更新 Predictor。

`--use-edge-relations` 显式开启 Predictor 的 `GATConv(edge_dim=7)`；
`--use-sampler-edge-relations` 独立开启 RL Sampler relation 打分。两者默认关闭，
self-loop relation 固定为零，Sampler relation 开关不允许用于非 RL sampler。

## PPIPredictor

- 输入特征经过线性层、LayerNorm、ReLU 和 Dropout。
- 图编码使用多层 GAT，并通过 residual/LayerNorm 稳定训练。
- 读出使用 `h_u+h_v`、`|h_u−h_v|` 和子图节点均值；`--readout attention`
  （默认 mean）在此之上**加性**追加目标锚定的 LinkAttention 摘要 `z`
  （RISE-DDI 机制：e1/e2 作为独立查询、GATv2 式门控、PPR 位置编码拼进
  key、共邻/单侧邻接层各用独立编码器、其余节点 PE=0），输出宽 3h→4h。
  PPR 在无标签全图拓扑上 forward-push 预计算（`src/ppr.py`，α/eps 可配），
  按目标惰性缓存；attention 模式的 `forward` 需要 `node_ids`（全局蛋白 id）。
  全图 KG 下 static k1 + attention 为当前项目最优（MacAUC 0.8105 /
  MacF1 0.6158，docs §15/§16）。
- 输出 7 维 logits；推理概率直接对 logits 取 sigmoid。
- 支持多图 batch。

## 评估

- 训练期间仅评估验证集。
- 训练结束后在验证 Macro-AUC 最佳状态上评估一次测试集。
- 报告 Macro/Micro ROC-AUC 和固定 0.5 阈值的 Macro/Micro F1。
- 测试集按两端节点相对训练节点集合的可见性报告 BS、ES、NS。
