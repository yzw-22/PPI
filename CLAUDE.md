# PPI 项目状态

本项目使用预计算的 ESM-2 蛋白质 embedding 进行 7 类 PPI 多标签预测。模型由全图知识图谱（KG）、REINFORCE 子图 Sampler 和 GAT Predictor 组成。

## 代码结构

- `src/ppi_graph.py`：加载数据、聚合标签并构建全图 KG 与 split 局部图。
- `src/sampler.py`：为每个目标 PPI 生成子图轨迹。
- `src/predictor.py`：使用 GAT 编码子图并输出 7 维 logits。
- `src/trainer.py`：交替更新 Sampler 和 Predictor。
- `src/train_shs27k.py`：训练、验证和测试入口，支持 SHS27k、SHS148k 和 STRING。
- `tests/`：图构建、采样、训练图选择和 RTG 测试。

## 数据与 split 约束

- 训练/验证/测试的采样 KG 为**全数据集全图**（`PPIGraph.build_full_graph()`：
  全部蛋白节点 + 全部 PPI 边 + 全部 embedding）；`build_graph(split)` 的
  split-local 构建保留，仅用于训练节点集合查询（测试 BS/ES/NS 分组）与外部使用。
- `edge_index` 使用局部节点索引，`node_index` 保存 local → global 映射，`node_feat` 与局部节点逐行对应。
- `edge_label` 是公共返回字段，保存每条 PPI 边的 7 维 multi-hot 标签。
- SHS27k 支持 `bfs/dfs/random`，SHS148k 支持 `dfs/random`（bfs 需
  `--root dataset_ppisplit`，默认 `dataset/` 不提供该文件），STRING 支持
  `dfs`；非法组合与缺失的 split 文件都在参数解析阶段报错。
- split 只决定各阶段的目标 PPI 对与训练节点集合；Sampler、代理候选、frontier 和 Predictor 输入全部使用全图的节点与边。
- 每次采样前从安全邻接中移除目标边 `(u,v)` 和 `(v,u)`，避免标签泄漏。
- `--use-edge-relations` 可选开启 relation-aware GAT：只有 train split 边携带
  7 维 multi-hot relation；val/test 拓扑边、虚拟 proxy 边和 GAT self-loop 的
  relation 恒为全零。查询目标边仍先被完全移除，因此自身标签不会进入输入。
- `--use-sampler-edge-relations` 独立开启 RL Sampler 的 relation-aware 动作打分：
  候选节点与当前已选节点间的可见关系按逐维 max/OR 聚合、投影并加到 candidate
  表示。该开关只适用于 `--sampler rl`，可与 Predictor relation 开关独立消融。

## 当前 Sampler 设计

- 只有安全邻接为空的目标才选择虚拟 proxy；proxy 从全图的非目标节点中按 ESM embedding 余弦相似度选择，两个目标可以共享 proxy。
- `baseline_graph` 是唯一初始图 `G_0`，只包含 `u`、`v` 和必要的虚拟 proxy；安全一跳邻居不预先采样。
- `G_0` 保留初始节点之间的安全诱导边和必要的虚拟 proxy 边；目标边不进入 `G_0` 或 step graph。
- 后续动作候选是当前 frontier，但只保留距 `G_0` 种子（`u`、`v`、proxy）不超过 `k_hops` 的安全节点；默认 `k_hops=1`。`max_steps` 只限制动作次数（默认 `10`），当前没有 STOP 动作或双目标平衡约束。
- 训练使用 Categorical 随机动作和可导 `log_prob`；评估及 Predictor 更新使用贪心动作。
- `StaticNeighborhoodSampler`（消融用、不可学习、零参数）：直接取 `G_0` 种子的
  全部安全 `k_hops` 邻居（区域诱导子图），轨迹无动作、仅训练 Predictor；用于
  检验 RL 选取的作用（static 图是任一 RL 轨迹图的信息上界）。
- `RandomSubsetSampler`（消融用、不可学习、零参数）：从 `k_hops` 区域均匀随机
  取与 RL 同规模（`min_size`~`max_size`，恒含 u/v）的节点子集 + 诱导边，轨迹
  无动作、仅训练 Predictor；与 RL 同规模对比，用于分离"上下文量"与"选取
  策略"。
- 动作 score 使用独立的 state/candidate 投影，拼接后映射回 hidden_dim 并加回
  投影 state（残差），再过 LN/Tanh 打分头：
  `Linear(2*hidden_dim→hidden_dim)（+state 残差）→ Linear(hidden_dim→hidden_dim//2)
  → LayerNorm → Tanh → Linear(hidden_dim//2→1)`。
- 开启 Sampler relation 后，candidate 投影会先加上候选到当前已选子图的 7 维
  relation 聚合投影；未知、held-out、目标边及 proxy relation 均为零。

## RL 与训练

每条轨迹的奖励和 return-to-go 为：

\[
r_t=L_{t-1}-L_t, \qquad G_t=r_t+\gamma G_{t+1}
\]

其中第一步的 \(L_{t-1}\) 是初始图 \(G_0\) 的 loss；不包含子图大小或
\(\Delta n\) 惩罚。

Sampler 更新为（无学习 baseline，advantage 即 return-to-go）：

\[
\hat A_t=\frac{A_t-\operatorname{mean}_{\mathrm{batch}}(A)}
{\max(\operatorname{std}_{\mathrm{batch}}(A),10^{-8})},
\qquad A_t=G_t
\]

\[
L_{policy}=-\log\pi(a_t|s_t)\operatorname{stopgrad}(\hat A_t),
\qquad L_{sampler}=L_{policy}
\]

- Sampler 更新时 Predictor 冻结，用 `G_0` 和所有 step graph 的 BCE loss 计算增量奖励；同一 batch 的全部动作 step 对 detached return-to-go 做标准化。
- Predictor 更新时 Sampler 冻结，只使用每条贪心轨迹的 `final_graph`；无动作时 `final_graph` 就是 `G_0`。
- Predictor 与 Sampler 的 relation-aware 模式分别由两个独立开关控制，默认均关闭
  以保持历史实验可复现。
- F1 使用固定阈值 `0.5`。
- 训练期间只评估验证集；按验证 Macro-AUC 保存最佳状态，训练结束后仅在最佳状态上测试一次。
- 指定 `--checkpoint-dir` 时保存最佳 checkpoint；未指定时将最佳 Sampler/Predictor 状态保存在内存中。

典型命令：

```bash
python -m src.train_shs27k \
  --dataset SHS27k \
  --split bfs \
  --device cuda \
  --epochs 10 \
  --hidden-dim 256 \
  --k-hops 1 \
  --max-steps 10 \
  --use-edge-relations \
  --reinforce-gamma 0.9
```

测试集额外按训练节点可见性分为 BS、ES、NS；空分组返回 `count=0` 和 `None` 指标。

## 性能优化方向

详见 [docs/README.md](docs/README.md)（docs 唯一报告，含性能优化方向与历史实验记录）。当前优先级为：缓存投影结果、增量维护状态和减少 Predictor 重复前向；STRING 扩展时再考虑 CSR tensor 邻接和 tensor frontier（全图下收益更大）。

## 提交约定

- **只运行实验、不修改代码时，不单独提交**：实验记录（`docs/README.md` 更新、
  结果 JSON 等）保留在工作区未提交状态；
- 直到下一次**代码修改**发生时，才连同累积的实验记录一并提交（可沿用
  src / docs 分组提交风格，但实验记录不单独成一次提交）。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```
