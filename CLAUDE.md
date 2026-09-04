# PPI 项目状态

本项目使用预计算的 ESM-2 蛋白质 embedding 进行 7 类 PPI 多标签预测。模型由全图知识图谱（KG）、REINFORCE 子图 Sampler 和 GAT Predictor 组成。

## 当前结论（截至 docs/README.md §15，2026-09-04）

- **最优配置**：`--sampler static --k-hops 1 --readout attention`（MacAUC
  0.8105 / MacF1 0.6158 / MicF1 0.6454，599s）——A1 读出重构（target 锚定
  LinkAttention + PPR 位置编码）是唯一抬高天花板的改动；
- **RL 小预算平台**：6.5–7 节点预算下 RL/heur 恒在 ~0.78，sampler 侧修复
  （结构特征 P2a、margin 奖励 R1、attention 读出 V2/V3）均无法突破；
- RL 实验默认带 `--reward-margin`（更稳、`mean_final_margin` 诊断免费）；
  heur 是 sampler 改动的行为基准（同预算配对判定，打不过 heur 即无效）；
- 未测路径：A2（static 底座 ∪ RL 增补，RISE-DDI 采样语义）。

## 代码结构

- `src/ppi_graph.py`：加载数据、聚合标签并构建全图 KG 与 split 局部图；
- `src/ppr.py`：无标签全图拓扑上的稀疏 PPR（forward-push，按目标惰性缓存）；
- `src/sampler.py`：RL Sampler 及 static / random-subset / heuristic 消融；
- `src/predictor.py`：GAT 编码 + 读出（mean / attention+PPR），7 维 logits；
- `src/trainer.py`：交替更新 Sampler 与 Predictor（BCE 差分或 margin 奖励）；
- `src/train_shs27k.py`：训练、验证和测试入口，支持 SHS27k、SHS148k 和 STRING；
- `tests/`：86 项单测（图构建、PPR、采样、奖励、读出与入参守卫）；
- 机制细节与不变量见 [src/CLAUDE.md](src/CLAUDE.md)，实验记录（唯一事实源）
  见 [docs/README.md](docs/README.md)。

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
- `--use-edge-relations`（Predictor GAT）与 `--use-sampler-edge-relations`
  （RL 动作打分）独立开关，默认关闭；只有 train split 边携带 7 维 multi-hot
  relation，val/test 拓扑边、虚拟 proxy 边和 self-loop 恒为全零。

## 当前 Sampler 设计

- 只有安全邻接为空的目标才选择虚拟 proxy；proxy 从全图的非目标节点中按 ESM
  embedding 余弦相似度选择，两个目标可以共享 proxy。
- `baseline_graph` 是唯一初始图 `G_0`，只包含 `u`、`v` 和必要的虚拟 proxy
  及其安全诱导边；目标边不进入 `G_0` 或 step graph。
- 后续动作候选是当前 frontier，只保留距 `G_0` 种子不超过 `k_hops`（默认 1）
  的安全节点；`max_steps` 只限制动作次数（默认 10），无 STOP 动作。
- 训练使用 Categorical 随机动作和可导 `log_prob`；评估及 Predictor 更新使用
  贪心动作。打分头/结构特征/relation 的公式细节见 src/CLAUDE.md。
- `StaticNeighborhoodSampler` / `RandomSubsetSampler` / `HeuristicSampler`
  （均不可学习、零参数、轨迹无动作）：分别取全部安全 k-hop 区域、同预算
  随机子集、同预算确定性拓扑排序（共邻→单侧→其余）；heur 为 RL 的行为基准。

## RL 与训练

- Sampler 更新：Predictor 冻结（eval + no_grad），现行奖励为增量 BCE 差分
  `r_t = L(G_{t−1}) − L(G_t)`（首步以 `G_0` 为前项，无 Δn 惩罚）；
  `--reward-margin`（默认关闭）改为固定 G0 参考的标签对齐平均概率边际改进
  `M(p)=mean_j((2y_j−1)·p_j) ∈ [−1,1]`，非对称缩放默认 2:1。
  return-to-go `G_t = r_t + γG_{t+1}`，advantage 为 batch 内标准化的
  detached RTG，`L_pol = −log π·stopgrad(Â)`。
- Predictor 更新：Sampler 冻结，只用每条贪心轨迹的 `final_graph` 做 BCE。
- F1 使用固定阈值 `0.5`；`mean_final_margin` 两种奖励模式均进入 epoch 记录。
- 训练期间只评估验证集；按验证 Macro-AUC 保存最佳状态（`--checkpoint-dir`
  落盘或内存保留），训练结束后仅在最佳状态上测试一次。

## 典型命令

```bash
# 当前最优（static + attention 读出，无 sampler 训练）
python -m src.train_shs27k --dataset SHS27k --split bfs --device cuda \
  --epochs 20 --hidden-dim 128 --sampler static --k-hops 1 --readout attention

# RL 基线（margin 奖励）
python -m src.train_shs27k --dataset SHS27k --split bfs --device cuda \
  --epochs 20 --hidden-dim 128 --sampler rl --max-steps 5 \
  --reinforce-gamma 0.9 --reward-margin
```

测试集额外按训练节点可见性分为 BS、ES、NS；空分组返回 `count=0` 和 `None` 指标。

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
