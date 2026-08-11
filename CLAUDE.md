# PPI 项目状态

本项目使用预计算的 ESM-2 3B 蛋白质嵌入（2560 维）进行 7 类 PPI 多标签预测。模型由 split-local PPI 图、REINFORCE 子图采样器和 GAT Predictor 组成。

数据说明见 [dataset/CLAUDE.md](dataset/CLAUDE.md)，核心实现说明见 [src/CLAUDE.md](src/CLAUDE.md)。

## 代码结构

- `src/ppi_graph.py`：加载数据、OR 聚合 7 类标签、构建 train/val/test 局部图。
- `src/sampler.py`：为每个目标 PPI 生成可变长度的 REINFORCE 子图轨迹。
- `src/predictor.py`：使用 GAT 编码子图并输出 7 维 PPI logits。
- `src/trainer.py`：交替更新 Sampler 和 Predictor。
- `src/train_shs27k.py`：SHS27k 训练、验证和测试入口，支持 `bfs`、`dfs`、`random` 划分。
- `tests/`：当前实现的图构建、采样安全性、final-graph 训练和 RTG 单元测试。

## 当前数据与索引约束

- 支持 SHS27k、SHS148k 和 STRING；训练入口使用 SHS27k，划分由 `--split` 指定（默认 `bfs`）。
- `PPIGraph.build_graph(split_name="train", undirected=True)` 始终返回当前 split 的局部节点图。
- `edge_index` 使用局部节点索引；`node_index` 保存 local → global 蛋白索引；`node_feat` 与局部节点逐行对应。
- 不存在 `remap_nodes` 可选项，split-local 是固定语义。
- Sampler、代理候选、frontier 扩展和 Predictor 输入都只能使用当前 split 的节点与边。
- 目标边 `(u,v)` 和 `(v,u)` 在每个目标采样前从安全邻接中移除，避免标签泄漏。

## 当前采样语义

- 目标节点只有在安全邻接为空时才会获得虚拟代理；代理从当前 split 的非目标节点中按余弦相似度选择，两个目标可以共享代理。
- `baseline_graph` 只含两个目标节点且无边。
- `initial_graph` 包含目标节点、必要的代理以及已选节点之间的真实安全边；虚拟边和真实边都提供给 Predictor，当前没有 edge type。
- 采样区域以 `initial_graph` 的全部节点为种子，在删除目标边后的安全图上计算 `k_hops` 邻域，默认 `k_hops=3`。
- 每一步的候选为当前 frontier 与该 3-hop 区域的交集；动作数最多为 `max_steps`。`max_steps` 是动作上限，不是 hop 深度，也没有 STOP 动作。
- 训练时使用 Categorical 随机动作并记录可导的 `log_prob`；评估和 Predictor 更新使用贪心动作。
- 当前未实现双目标平衡扩展；k-hop 限制本身不能保证两个目标处于同一连通分量或都获得邻居。

## 当前 RL 与训练语义

对每条轨迹的第 `t` 步：

\[
r_t=L_{baseline}-L_t
\]

\[
G_t=r_t+\gamma G_{t+1}
\]

其中 `G_t` 是 return-to-go，默认 `gamma=1.0`，实验可通过 `--reinforce-gamma` 设置为 `0.95`。Sampler 更新为：

\[
L_{policy}=-\log\pi(a_t|s_t)\,\operatorname{stopgrad}(G_t-V(s_t))
\]

\[
L_{value}=\operatorname{MSE}(V(s_t),G_t),
\qquad
L_{sampler}=L_{policy}+\beta L_{value}
\]

- Sampler 更新时 Predictor 参数冻结，Predictor 对 baseline graph 和所有 step graph 计算 BCE loss 以产生 reward。
- reward、return 和 Sampler loss 对所有 trajectory step 等权平均；较长轨迹因此贡献更多 step 项。
- Predictor 更新时 Sampler 参数冻结，以贪心动作生成轨迹，只使用 `final_graph`；无动作时使用含代理的 `initial_graph`。
- Predictor 使用 BCE with logits；目标对 readout 使用 `h_u+h_v`、`|h_u-h_v|` 和子图节点均值，端点交换保持不变。
- Sampler 使用独立的 `Linear(esm_dim→hidden_dim)` 分别将当前子图均值和每个候选邻居投影到 hidden space；拼接两者后输入 pairwise MLP（`Linear(2*hidden_dim→action_hidden) → LeakyReLU → Linear(action_hidden→1)`），再对候选动作做 softmax。MLP Linear 权重使用 Xavier uniform 初始化，偏置为 0，LeakyReLU 负斜率为 0.2。将 LeakyReLU 逐元素置于点积之前会使 state 项对候选统一、在 softmax 中抵消，因此已移除该打分方式。

## 训练与评估接口

训练入口参数的源码默认值与实验配置分开：当前 CLI 默认是 hidden 256、GAT 2 层、`max_steps=3`、`k_hops=3`、gamma 1.0；复现实验必须显式传入完整配置。

典型实验命令：

```bash
python -m src.train_shs27k \
  --split random \
  --epochs 10 \
  --hidden-dim 512 \
  --max-steps 10 \
  --k-hops 3 \
  --reinforce-gamma 0.95 \
  --gnn-layers 3
```

每个 epoch 报告训练 Sampler loss、Predictor loss、mean reward、验证集指标和测试集指标。验证/测试指标包括 Macro/Micro ROC-AUC 与 F1。F1 始终使用固定 0.5 阈值，与同类论文的报告约定保持一致；不在验证集或测试集上调节阈值。

测试 PPI 还按两端节点是否出现在训练图中分组：

- BS（both seen）：两端都在训练节点集合中；
- ES（either seen）：恰好一端在训练节点集合中；
- NS（neither seen）：两端都不在训练节点集合中。

空分组报告 `count=0` 和 `None` 指标；当某类别在子集中只有单一取值时，Macro-AUC 不可定义，同样报告 `None` 而不是 NaN（此时验证集最佳 checkpoint 选择会跳过该 epoch）。SHS27k/bfs 测试集当前 BS 为 0，这是该 split 的结构性结果。

训练入口通过 `--checkpoint-dir DIR`（可选）启用验证集最佳 checkpoint：每个 epoch 结束时若验证 Macro-AUC 刷新最优，保存 `best_{epoch}.pt`（Sampler/Predictor 及各自优化器状态）；训练结束后加载该 checkpoint 回放一次测试集，结果写入输出 JSON 的 `best_checkpoint_test`，便于第三方按"最佳验证 epoch"复核。不传该参数则不保存、不回放，但输出仍包含 `best_epoch`/`best_val_macro_auc`。

## 历史基准结果

以下是改用当前 attention sampler 之前的历史结果，仅用于后续消融对照，
不是当前实现的结果。配置为 `hidden_dim=512`、`max_steps=10`、`k_hops=3`、
`reinforce_gamma=0.95`、3 层 GAT、seed 42；每个划分按验证集 Macro-AUC
选择 Epoch，再报告对应测试集结果。

| 划分 | Q/K | train/val/test PPI | Epoch | 最佳验证 Macro-AUC | 测试 Macro-AUC | 测试 Micro-AUC | 测试 Macro-F1 | 测试 Micro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BFS | 双层 MLP | 4562/1524/1538 | 9/10 | 0.7413 | 0.7173 | 0.7295 | 0.4763 | 0.5004 |
| DFS | 单层线性 | 4571/1524/1529 | 9/10 | 0.8012 | 0.8070 | 0.8370 | 0.5909 | 0.6473 |
| random | 单层线性 | 4574/1525/1525 | 10/10 | 0.8743 | 0.8809 | 0.8952 | 0.6552 | 0.7034 |

历史结果显示 random 的整体性能最高，DFS 次之，BFS 最低。主要原因之一是
划分结构不同：BFS/DFS 测试集均为 BS=0、ES/NS 分布为
DFS 1215/314、BFS 1078/460；random 则为 BS/ES/NS=1377/138/10。
因此整体指标不能脱离节点可见性分布直接比较。DFS 第 10 Epoch 的
Predictor loss 为 0.2299、Sampler loss 为 0.1894，但其验证性能已从第 9
Epoch 回落，正式使用时应保存验证集最佳 checkpoint。

此前独立 `state_proj/neighbor_proj` 线性 attention sampler 已完成 SHS27k/BFS
10 Epoch 训练，现作为 pairwise MLP 改造前基线。配置为 `hidden_dim=512`、
`max_steps=10`、`k_hops=3`、`reinforce_gamma=0.95`、3 层 GAT、seed 42。
第 9 Epoch 达到最佳验证
Macro-AUC 0.7266；对应测试 Macro-AUC/Micro-AUC 为 0.7192/0.7530，
Macro-F1/Micro-F1 为 0.4974/0.5432。测试 ES（1078 对）AUC 为
0.7307/0.7657、F1 为 0.5175/0.5629；NS（460 对）AUC 为
0.6974/0.7221、F1 为 0.4447/0.4929；BS 样本数为 0。

第 10 Epoch 验证 Macro-AUC 降至 0.6963，而 Predictor loss 继续从第 9
Epoch 的 0.2262 降至 0.2127，显示后期存在过拟合或 REINFORCE 采样波动。
当前结果仍是单 seed，需多 seed 和严格 checkpoint 对照后才能判断 attention
打分相对历史 Query/Key 方案的稳定收益。

当前 pairwise MLP sampler 已完成同配置的 SHS27k/BFS 10 Epoch 训练：
`hidden_dim=512`、`max_steps=10`、`k_hops=3`、`reinforce_gamma=0.95`、
3 层 GAT、seed 42。第 9 Epoch 验证 Macro-AUC 最佳，为 `0.7395`；对应
验证 Micro-AUC/Macro-F1/Micro-F1 为 `0.7563/0.5070/0.5576`。对应测试集
Macro-AUC/Micro-AUC/Macro-F1/Micro-F1 为
`0.7347/0.7607/0.5192/0.5629`。测试 ES（1078 对）为
`0.7471/0.7757/0.5455/0.5853`，NS（460 对）为
`0.7045/0.7224/0.4542/0.5078`，BS 为 0 对。

该实验每个 epoch 的训练损失如下；Predictor loss 持续下降，但验证性能在
第 9 Epoch 后明显回落，仍表现出交替训练中的过拟合或 REINFORCE 波动：

| Epoch | Sampler loss | Predictor loss | mean reward | Val Macro-AUC |
|---:|---:|---:|---:|---:|
| 1 | -0.0119 | 0.5236 | 0.0154 | 0.6413 |
| 2 | 0.0472 | 0.4346 | 0.0451 | 0.7211 |
| 3 | 0.0550 | 0.3800 | 0.0560 | 0.6774 |
| 4 | 0.0615 | 0.3399 | 0.0699 | 0.7363 |
| 5 | 0.0841 | 0.3048 | 0.0809 | 0.7091 |
| 6 | 0.0853 | 0.2785 | 0.0871 | 0.7299 |
| 7 | 0.1153 | 0.2581 | 0.1007 | 0.6861 |
| 8 | 0.1289 | 0.2419 | 0.1053 | 0.7139 |
| 9 | 0.1508 | 0.2345 | 0.1146 | 0.7395 |
| 10 | 0.1409 | 0.2149 | 0.1249 | 0.6898 |

实验原始结果保存在 `/tmp/shs27k_bfs_steps10_gamma095_gat3_pairwise_mlp.json`，
未纳入项目代码库。该结果是单 seed，不能单独证明 pairwise MLP 相对其他
sampler 架构的稳定优势。

为检查随机性，使用相同配置更换为 seed 123 再训练 10 Epoch。最佳验证
Macro-AUC 出现在 Epoch 10，为 `0.7347`；对应验证
Micro-AUC/Macro-F1/Micro-F1 为 `0.7506/0.4797/0.5365`。对应测试集
Macro-AUC/Micro-AUC/Macro-F1/Micro-F1 为
`0.7216/0.7452/0.4732/0.5184`。测试 ES（1078 对）为
`0.7269/0.7528/0.4929/0.5357`，NS（460 对）为
`0.7205/0.7259/0.4233/0.4733`，BS 仍为 0 对。原始结果保存在
`/tmp/shs27k_bfs_steps10_gamma095_gat3_pairwise_mlp_seed123.json`。

两个 seed 的最佳验证 Macro-AUC 为 `0.7395/0.7347`，对应测试
Macro-AUC 为 `0.7347/0.7216`；最佳 Epoch 分别为 9 和 10，说明当前
REINFORCE 训练仍有明显随机波动，后续应报告多 seed 均值和标准差。

## 已知问题与后续方向

- 训练脚本每个 epoch 都评估测试集；严格实验应只用验证集选择 checkpoint，最后测试一次。已通过 `--checkpoint-dir` 支持按验证集 Macro-AUC 保存最佳 checkpoint 并在训练结束后回放一次测试集；测试集的逐 epoch 输出仍保留。
- Sampler 的 Python set 邻接、重复 tensor 构造、代理相似度扫描和候选投影仍是 STRING 上的性能瓶颈。
- 当前 pairwise MLP sampler 仅完成 BFS seed=42 和 seed=123 两次实验；候选节点逐一拼接并经过 MLP，单次实验约 25 分钟，仍需关注运行时和更多 seed 的稳定性。
- Sampler 仍可能偏向某一目标侧；k-hop 限制只限制区域，不提供双目标平衡保证。
- 没有 STOP 动作，达到 frontier 为空或动作上限才结束；长轨迹会增加计算量和 step loss 权重。
- F1 按同类论文惯例使用固定 0.5 阈值；类别不均衡尚未通过 class weight 等训练策略处理，不进行验证集阈值校准。
- NS 子集在不同划分中的样本量差异很大；样本过少或某类别只有单一取值时，Macro-AUC 不可定义。
- 当前 pairwise MLP 已完成两个随机种子，但尚未有足够 seed 计算稳定置信区间或完成严格消融；历史表中的不同 sampler 架构不能直接进行跨 split 的纯性能比较。详细实验汇总见 [experiments/SHS27K_BFS_pairwise_MLP.md](experiments/SHS27K_BFS_pairwise_MLP.md)。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```
