# PPI 项目状态

本项目使用预计算的 ESM-2 3B 蛋白质嵌入（2560 维）进行 7 类 PPI 多标签预测。模型由 split-local PPI 图、REINFORCE 子图采样器和 GAT Predictor 组成。

数据说明见 [dataset/CLAUDE.md](dataset/CLAUDE.md)，核心实现说明见 [src/CLAUDE.md](src/CLAUDE.md)。

## 代码结构

- `src/ppi_graph.py`：加载数据、OR 聚合 7 类标签、构建 train/val/test 局部图。
- `src/sampler.py`：为每个目标 PPI 生成可变长度的 REINFORCE 子图轨迹。
- `src/predictor.py`：使用 GAT 编码子图并输出 7 维 PPI logits。
- `src/trainer.py`：交替更新 Sampler 和 Predictor。
- `src/train_shs27k.py`：SHS27k/bfs 训练、验证和测试入口。
- `tests/`：当前实现的图构建、采样安全性、final-graph 训练和 RTG 单元测试。

## 当前数据与索引约束

- 支持 SHS27k、SHS148k 和 STRING；训练入口当前使用 SHS27k/bfs。
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

## 训练与评估接口

训练入口参数的源码默认值与实验配置分开：当前 CLI 默认是 hidden 256、GAT 2 层、`max_steps=3`、`k_hops=3`、gamma 1.0；复现实验必须显式传入完整配置。

典型实验命令：

```bash
python -m src.train_shs27k \
  --epochs 6 \
  --hidden-dim 512 \
  --max-steps 20 \
  --k-hops 3 \
  --reinforce-gamma 0.95 \
  --gnn-layers 3
```

每个 epoch 报告训练 Sampler loss、Predictor loss、mean reward、验证集指标和测试集指标。验证/测试指标包括 Macro/Micro ROC-AUC 与 F1；F1 当前固定使用 0.5 阈值。

测试 PPI 还按两端节点是否出现在训练图中分组：

- BS（both seen）：两端都在训练节点集合中；
- ES（either seen）：恰好一端在训练节点集合中；
- NS（neither seen）：两端都不在训练节点集合中。

空分组报告 `count=0` 和空指标。SHS27k/bfs 测试集当前 BS 为 0，这是该 split 的结构性结果。

## 最近实验进展

配置均为 hidden 512、`k_hops=3`、gamma 0.95、GAT 3 层、6 Epoch、seed 42：

| max_steps | 最佳验证 Macro-AUC（epoch） | 对应测试 Macro-AUC | 对应测试 Micro-AUC | 平均最终节点数（抽样样本） |
|---:|---:|---:|---:|---:|
| 10 | 0.7219（3） | 0.7031 | 0.7299 | 12.33 |
| 20 | 0.7258（6） | 0.7111 | 0.7159 | 22.33 |

20 步显著增加子图规模、边数、环数和直径，但仍不能消除单侧扩展；测试样本中仍可能有一个目标孤立。单个 seed、不同最佳 epoch 和 F1 波动意味着不能据此断言 20 步稳定优于 10 步。

子图图像和结构摘要保存在：

- [subgraph_plots_k3_gat3](subgraph_plots_k3_gat3/)
- [subgraph_plots_k3_gat3_steps20](subgraph_plots_k3_gat3_steps20/)

## 已知问题与后续方向

- 训练脚本每个 epoch 都评估测试集，严格实验应只用验证集选择 checkpoint，最后测试一次；当前也没有 checkpoint 保存功能。
- Sampler 的 Python set 邻接、重复 tensor 构造、代理相似度扫描和候选投影仍是 STRING 上的性能瓶颈。
- Sampler 仍可能偏向某一目标侧；k-hop 限制只限制区域，不提供双目标平衡保证。
- 没有 STOP 动作，达到 frontier 为空或动作上限才结束；长轨迹会增加计算量和 step loss 权重。
- F1 使用固定阈值，类别不均衡尚未通过 class weight 或验证集阈值校准处理。
- 当前结果只有单个随机种子，尚未进行多 seed 置信区间和严格消融。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```
