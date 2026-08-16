# PPI 项目状态

本项目使用预计算的 ESM-2 蛋白质 embedding 进行 7 类 PPI 多标签预测。模型由 split-local PPI 图、REINFORCE 子图 Sampler 和 GAT Predictor 组成。

## 代码结构

- `src/ppi_graph.py`：加载数据、聚合标签并构建 train/val/test 局部图。
- `src/sampler.py`：为每个目标 PPI 生成 learned 子图轨迹，并提供固定 seed 的随机一跳和随机逐步扩展消融 Sampler。
- `src/predictor.py`：使用 GAT 编码子图并输出 7 维 logits。
- `src/trainer.py`：交替更新 Sampler 和 Predictor。
- `src/train_shs27k.py`：训练、验证和测试入口，支持 SHS27k、SHS148k 和 STRING。
- `tests/`：图构建、采样、训练图选择和 RTG 测试。

实验报告位于 `docs/`：BFS 主实验、gamma/复杂度惩罚消融和 main 分支基线均有独立报告。

## 数据与 split 约束

- `PPIGraph.build_graph()` 始终返回当前 split 的局部节点图。
- `edge_index` 使用局部节点索引，`node_index` 保存 local → global 映射，`node_feat` 与局部节点逐行对应。
- `edge_label` 是公共返回字段，保存每条 PPI 边的 7 维 multi-hot 标签。
- SHS27k 支持 `bfs/dfs/random`，SHS148k 支持 `dfs/random`，STRING 支持 `dfs`；非法组合在参数解析阶段报错。
- Sampler、代理候选、frontier 和 Predictor 输入只能使用当前 split 的节点与边。
- 每次采样前从安全邻接中移除目标边 `(u,v)` 和 `(v,u)`，避免标签泄漏。

## 当前 Sampler 设计

- 只有安全邻接为空的目标才选择虚拟 proxy；proxy 从当前 split 的非目标节点中按 ESM embedding 余弦相似度选择，两个目标可以共享 proxy。
- `baseline_graph` 是唯一初始图 `G_0`。对 `u`、`v` 和实际存在的 proxy，分别从各自 split 内安全一跳邻居中最多采样 `fixed_num` 个节点；默认 `fixed_num=1`。
- 一跳采样使用独立固定 seed `42`，重复节点合并去重，因此同一输入的 `G_0` 可复现。
- `G_0` 保留所有已选节点之间的安全诱导边和必要的虚拟 proxy 边；目标边不进入 `G_0` 或 step graph。
- 后续动作候选是当前 frontier，不使用 hop 限制；策略最多执行 `max_steps` 次决策，默认 `max_steps=10`。`fixed_num` 只限制每个实际初始 seed 的一跳采样数量，`max_steps` 只限制 action 次数；最终图大小取决于安全邻接、proxy 选择、节点去重和 frontier 候选，不使用派生的聚合上下文节点上限。策略包含显式 STOP 动作。
- 训练使用 Categorical 随机动作和可导 `log_prob`；评估及 Predictor 更新使用贪心动作。
- 动作 score 使用独立的 state/candidate 投影和 pairwise MLP：
  `Linear(2*hidden_dim→action_hidden) → LeakyReLU → Linear(action_hidden→1)`。
- `RandomSubgraphSampler` 用于消融：从 `{u, v, proxy}` 的安全一跳邻居并集中最多采样 10 个节点，seed 固定为 `42`；随机图直接作为 `final_graph`，不产生动作。
- `RandomIterativeSubgraphSampler` 复用 learned 的 `G0`，再从当前 frontier 每次随机扩展一个节点，最多扩展 10 次；无可学习参数，仅用于 Predictor 训练。
- 当前 CLI 的合法 `--sampler-mode` 只有 `learned`、`target_only`、`target_proxy`、`random_1hop10` 和 `random_iterative10`；不存在旧的 `random` 模式。

## RL 与训练

每条轨迹使用相邻图之间的增量奖励和 return-to-go。普通扩展动作的奖励为：

\[
r_t=L_{t-1}-L_t-\lambda\Delta n_t
\]

其中 \(\Delta n_t\) 是本步新增的非 baseline 节点数；显式 DONE/STOP 动作的奖励为 \(r_t=0\)。因此 DONE 不会重复获得前一步图的收益。return-to-go 为：

\[
G_t=r_t+\gamma G_{t+1}
\]

Sampler 更新为：

\[
L_{policy}=-\log\pi(a_t|s_t)\operatorname{stopgrad}(G_t-V(s_t))
\]

\[
L_{value}=\operatorname{MSE}(V(s_t),G_t), \qquad
L_{sampler}=L_{policy}+\beta L_{value}
\]

- Sampler 更新时 Predictor 冻结，用 `G_0` 和所有 step graph 的 BCE loss 计算奖励。
- Predictor 更新时 Sampler 冻结，只使用每条贪心轨迹的 `final_graph`；无动作时 `final_graph` 就是 `G_0`。
- F1 使用固定阈值 `0.5`。
- 训练期间只评估验证集；按验证 Macro-AUC 保存最佳状态，训练结束后仅在最佳状态上测试一次。
- 指定 `--checkpoint-dir` 时保存最佳 checkpoint；未指定时将最佳 Sampler/Predictor 状态保存在内存中。

典型命令：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --sampler-mode learned \
  --device cuda \
  --epochs 10 \
  --hidden-dim 256 \
  --fixed-num 1 \
  --max-steps 10 \
  --reinforce-gamma 0.9
```

固定基线模式为 `target_only`、`target_proxy`、`random_1hop10` 和 `random_iterative10`；这些模式均跳过 Sampler 更新，仅训练 Predictor。

`run.sh` 默认执行 BFS 的五组实验；DFS 的五组命令和已完成实验结果见实验报告。

测试集额外按训练节点可见性分为 BS、ES、NS；空分组返回 `count=0` 和 `None` 指标。

## 性能优化方向

详见 [TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md)。当前优先级为：缓存 `G_0` 和投影结果、增量维护状态、减少 adjacency 复制和 Predictor 重复前向；STRING 扩展时再考虑 CSR tensor 邻接和 tensor frontier。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```
