# 训练时间瓶颈与优化方案

> 性能优化方向。实验结论与调参记录见 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)。

## 现状

训练时间主要来自：Sampler 轨迹生成、Sampler 更新阶段对多个 step graph 的
Predictor 前向、验证集推理。`reinforce_gamma` 不影响运行时间。

## 训练稳定性约束（与速度无关，必须保持）

奖励 `r_t=L(G_{t-1})-L(G_t)`（第一步以 G0 为前项），无子图大小惩罚；batch
内 detached advantage 标准化后算 policy loss，value loss 回归未标准化
return-to-go。categorical/log-prob、return-to-go、advantage、value loss
保持 FP32。

## 主要瓶颈

1. **Sampler/Predictor 更新重复计算**：每 batch 先生成 stochastic trajectory
   并对 G0+所有 step graph 算 Predictor loss，再重新生成 greedy trajectory 做
   Predictor 更新；`max_steps=10` 时每 batch 最多约 320 个 step graph。
2. **验证/测试推理**：每 epoch 全量遍历验证集（greedy Sampler+Predictor）；
   测试只在最佳 checkpoint 评估一次，无逐轮开销。
3. **Sampler 的 Python 与动态 tensor 操作**：每 target 每步的 `set/list`、
   `sorted(frontier)`、动态构造 `selected/candidate_tensor`、`_make_graph()`
   的 dict/边列表；`int(action)` 可能触发 GPU→CPU 同步。
4. **重复的候选投影/状态均值/代理扫描**：候选节点重复过 `neighbor_proj`；
   每步重新索引 selected 求均值；`_nearest_proxy()` 每次全量归一化+扫描；
   G0 在更新/验证/测试中重复构造。
5. **邻接表复制（已消除）**：目标边懒惰屏蔽 view + 每 trajectory 一次 k-hop
   局部 BFS，禁止每步重复 BFS 与全数据集无界区域缓存。

## 优先级方案

| 优先级 | 方案 | 预期收益 | 风险 |
|---|---|---|---:|
| 已完成 | 训练期间取消逐 epoch 测试，只测最佳 checkpoint | 很高 | 很低 |
| 已完成 | 目标边懒惰屏蔽 view + 每 trajectory 一次 k-hop 区域 | STRING 中高 | 低 |
| P1 | 缓存 split-level 归一化 embedding 与 candidate projection | 高 | 低 |
| P1 | 增量维护 selected embedding sum/count | 中等 | 低 |
| P1 | 减少/合并 Predictor 重复 forward | 很高 | 中 |
| P2 | 评估用 `torch.inference_mode()` | 中等 | 很低 |
| P2 | 增大 train/eval batch size | 中等 | 中 |
| P2 | Predictor BF16/AMP | 中高 | 中 |
| P3 | CSR tensor 邻接 + tensor frontier | STRING 高 | 较高 |

## 推荐实施顺序

1. **低风险**：`inference_mode()`；缓存归一化 embedding；trajectory 内预计算
   `neighbor_proj` 只索引候选；`state_sum/count` 增量均值。
2. **Sampler 图操作**：减少动态 tensor 与 Python dict/边列表；避免
   `int(action)` 同步；缓存/批量构造 graph index。
3. **GPU**：batch size 32/64/128 与 eval 64/128/256 扫描；Predictor GAT 优先
   BF16 autocast，Sampler 数值保持 FP32；比较速度/显存/NaN/reward/验证指标。
4. **STRING**：CSR 邻接（`row_ptr/col_index`）、tensor visited/frontier、
   批量候选投影与评分、split-local remap 缓存降全局扫描。

## Profiling 要求

优化前后分别统计：`sampler_g0_build`、`sampler_action_score`、
`sampler_graph_build`、`predictor_baseline`、`predictor_step_graphs`、
`predictor_update`、`validation`、`test_evaluation`。CUDA 异步执行，测 GPU
时间前 `torch.cuda.synchronize()`；建议 `torch.profiler` 同时记录
CPU/CUDA 活动、算子形状与显存。

## 必须保持的不变量

- 节点/代理/边均来自当前 split；目标边始终从安全邻接、G0 与 step graph 移除；
- G0 只含 u/v/proxy；candidate 在一次性安全 `k_hops` 区域内；G0 保留选中节点
  间全部安全诱导边；
- Predictor 训练与评估使用 `final_graph`；REINFORCE 增量奖励 + 按 gamma 的
  return-to-go；F1 阈值固定 0.5；
- 推理特征只能来自 ESM embedding 与无标签拓扑（禁止边标签/测试标签）。
