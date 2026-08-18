# 训练时间瓶颈与优化方案

## 现状

训练时间主要来自 Sampler 轨迹生成、Sampler 更新阶段对多个 step graph 的 Predictor 前向计算，以及验证集推理；`reinforce_gamma` 本身不会显著影响运行时间。

## 训练稳定性约束

Sampler 的普通动作奖励使用相邻子图的 Predictor loss 改善：

\[
r_t=L(G_{t-1})-L(G_t)
\]

第一步中的 \(G_{t-1}\) 是 \(G_0\)。当前实现不对新增节点数或子图大小施加
惩罚。每个 sampler batch 汇总全部动作的 detached advantage
\(A_t=G_t-V(s_t)\)，并按 batch 均值和总体标准差标准化后计算 policy loss；
value loss 仍使用未标准化的 return-to-go。该机制用于降低 REINFORCE 更新方差，
不是运行时加速。categorical/log-prob、return-to-go、advantage 和 value loss 保持
FP32。

## 主要瓶颈

### 1. Sampler 更新和 Predictor 更新重复计算

每个训练 batch 先生成 stochastic trajectory，并对 `G_0` 和所有 step graph 计算 Predictor loss；随后又重新生成 greedy trajectory，并对 final graph 执行 Predictor 更新。`max_steps=10` 时，一个 batch 最多会产生约 320 个 step graph，GAT 前向计算成本较高。

### 2. 验证集推理和最终测试

训练循环每轮完整遍历验证 PPI，并重新执行 greedy Sampler 和 Predictor 推理；训练结束后再对最佳验证状态执行一次测试。测试集不参与逐轮模型选择，因此不会产生额外的逐轮测试开销。

### 3. Sampler 的 Python 和动态 tensor 操作

每个 target、每一步会执行 Python `set`/`list` 更新、`sorted(frontier)`、动态构造 `selected_tensor` 和 `candidate_tensor`，并在 `_make_graph()` 中构造 Python 字典和边列表。`int(action)` 还可能触发 GPU 到 CPU 的同步。

### 4. 重复的候选投影、状态均值和代理扫描

- 同一 trajectory 中候选节点可能重复经过 `neighbor_proj`；
- 每一步都重新索引 selected nodes 并计算 embedding 均值；
- `_nearest_proxy()` 每次都重新归一化整个节点 embedding 矩阵并扫描余弦相似度；
- 同一个 PPI 的 `G_0` 在 Sampler 更新、Predictor 更新、验证和测试中重复构造。

### 5. 每个 target 的邻接表复制（已消除）

共享邻接表通过只覆盖两个目标节点的只读 view 排除目标边，不复制完整
`adjacency` 列表。k-hop 可达区域在每条 trajectory 初始化时做一次局部 BFS，
action step 仅增量维护受限 frontier；不能在每步重复 BFS，也不应为全数据集
target 建立无界区域缓存。SHS27k 影响有限，但在 STRING 上可避免明显的 O(N)
邻接复制和不可控缓存占用。

## 优先级方案

| 优先级 | 方案 | 预期收益 | 风险 |
|---|---|---:|---:|
| 已完成 | 训练期间取消逐 epoch 测试，训练结束只测试最佳 checkpoint | 很高 | 很低 |
| P1 | 缓存 split-level 归一化 embedding 和 candidate projection | 高 | 低 |
| P1 | 增量维护 selected embedding sum/count | 中等 | 低 |
| P1 | 减少或合并 Predictor 的重复 forward | 很高 | 中 |
| 已完成 | 使用目标边懒惰屏蔽 view，并在每条 trajectory 只构造一次 k-hop 区域 | STRING 上中等到高 | 低 |
| P2 | 评估阶段使用 `torch.inference_mode()` | 中等 | 很低 |
| P2 | 增大 train/eval batch size | 中等 | 中 |
| P2 | Predictor 尝试 BF16/AMP | 中等到高 | 中 |
| P3 | CSR tensor 邻接和 tensor frontier | STRING 上高 | 较高 |

## 推荐实施顺序

### 第一阶段：低风险

1. 已完成：训练过程中只评估验证集；根据验证 Macro-AUC 保存最佳状态，训练结束后只测试一次。
2. 验证和测试使用 `torch.inference_mode()`。
3. 缓存 split-level 归一化 embedding。
4. 在一次 Sampler trajectory 内预先计算 `neighbor_proj(node_features)`，每步只索引候选节点。
5. 用 `state_sum / state_count` 增量维护状态均值。

### 第二阶段：Sampler 图操作

1. 已完成：用共享 adjacency 的目标边懒惰屏蔽 view 替代 `list(adjacency)`，并仅在 trajectory 初始化时构造 k-hop 区域。
2. 减少动态 tensor 构造和 Python 字典/边列表构造。
3. 避免 `int(action)` 导致的 GPU 同步。
4. 缓存或批量构造 graph index。

### 第三阶段：GPU 优化

1. 在显存允许范围内测试 train batch size `32/64/128` 和 eval batch size `64/128/256`。
2. 优先对 Predictor GAT 使用 BF16 autocast。
3. 保持 categorical/log-prob、return-to-go、标准化 advantage 和 value loss 使用 FP32；再单独评估 Sampler projection/MLP 是否适合 BF16。
4. 比较速度、显存、NaN 情况、reward 和验证性能。

### 第四阶段：STRING 扩展

1. 将 Python set adjacency 改为 CSR 格式 `row_ptr/col_index`。
2. 使用 tensor 化 visited mask 和 frontier。
3. 对候选节点进行批量投影和批量动作评分。
4. 结合 split-local remap 和缓存降低全局节点扫描。

## Profiling 要求

优化前后应分别统计：

- `sampler_g0_build`
- `sampler_action_score`
- `sampler_graph_build`
- `predictor_baseline`
- `predictor_step_graphs`
- `predictor_update`
- `validation`
- `test_evaluation`

CUDA 是异步执行的，测量 GPU 时间前应调用 `torch.cuda.synchronize()`；建议使用 `torch.profiler` 同时记录 CPU/CUDA 活动、算子形状和显存。

## 必须保持的不变量

- 所有节点、代理和边都来自当前 train/val/test split；
- 目标边始终从安全邻接、`G_0` 和 step graph 中移除；
- `G_0` 只包含 `u/v/proxy`；candidate 节点始终位于从这些种子计算的一次性安全 `k_hops` 区域内；
- `G_0` 保留选中节点之间的全部安全诱导边；
- Predictor 训练和评估继续使用 `final_graph`；
- REINFORCE 使用相邻图的增量奖励，return-to-go 继续按既定的 `gamma` 从后向前计算；
- F1 阈值固定为 0.5。
