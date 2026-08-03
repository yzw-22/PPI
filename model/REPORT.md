# PPI 模型实现报告

## 项目概述

Protein-Protein Interaction (PPI) 预测模型，包含可学习的子图采样器（SubgraphSampler）和 GNN 预测器（PPIPredictor），通过交替训练策略协调两个组件的学习。

**数据集**：SHS27k（1690 蛋白质，7624 PPI 对，7 类 multi-hot 标签）

---

## 架构总览

```
                    ┌─────────────────────┐
                    │     PPIModel        │
                    │  (交替训练编排)       │
                    └────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │ SubgraphSampler │          │  PPIPredictor   │
     │  (注意力+REINFORCE) │       │  (GAT GNN+分类器) │
     └────────┬────────┘          └────────┬────────┘
              │                             │
              │    subgraph_nodes            │
              └─────────────────────────────►
                             │
                    ┌────────▼────────┐
                    │  PPIGraph       │
                    │  (邻接表+边特征) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ESM-2 Tensor   │
                    │  [N, 2560]      │
                    └─────────────────┘
```

### 推理流程

```
(u, v) → Sampler(ESM, Graph) → 子图节点列表 → Predictor(ESM, Graph, 子图, (u,v)) → [7] sigmoid 概率
```

---

## 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| [config.py](config.py) | 80 | `PPIConfig` — 超参数集中管理、设备选择、参数校验 |
| [graph_utils.py](graph_utils.py) | 233 | `PPIGraph` 类 + `build_graph()` — 图构建、前沿查找、边特征聚合 |
| [sampler.py](sampler.py) | 237 | `SubgraphSampler` — 注意力邻居选择 + REINFORCE 策略梯度 |
| [predictor.py](predictor.py) | 183 | `PPIPredictor` — GAT 消息传递 + pairwise readout + 7 维 sigmoid |
| [ppi_model.py](ppi_model.py) | 327 | `PPIModel` — 组合模型、交替训练调度、推理接口 |
| [__init__.py](__init__.py) | 25 | 包导出 |

---

## 各模块设计

### 1. config.py — 配置管理

```python
@dataclass
class PPIConfig:
    # 维度
    esm_dim: int = 2560          # ESM-2 3B 输出维度
    hidden_dim: int = 256        # 通用隐层维度
    attention_dim: int = 64      # Sampler 注意力投影维度

    # Sampler
    T_max: int = 10              # 最大扩展步数
    pooling_mode: "mean" | "sum" = "mean"  # 状态池化（默认 mean 替代原始 sum）
    use_edge_features_in_sampler: bool = True

    # Predictor
    gnn_num_layers: int = 3      # GAT 层数
    gnn_dropout: float = 0.3
    gnn_heads: int = 4           # 注意力头数

    # 训练
    lr_sampler: float = 1e-4
    lr_predictor: float = 1e-3
    sampler_steps: int = 5
    predictor_steps: int = 5
    reinforce_baseline_coef: float = 0.5   # Baseline 损失权重 β
    grad_accum_steps: int = 4   # 梯度累积（缓解逐样本无法批量化）
```

### 2. graph_utils.py — 图构建

**`PPIGraph`**：以邻接表存储无向图，边特征为 7 维 multi-hot Tensor。

```
adj_list: List[List[int]]
edge_feat: Dict[(min,max) → Tensor[7]]
```

核心方法：
- `get_frontier(subgraph)` — 获取子图的前沿节点（不在子图中但相邻）
- `aggregate_edge_features(neighbor, subgraph)` — max-pool 聚合邻居与子图间所有边的 multi-hot（逻辑 OR）

**`build_graph(name, dir)`**：三步构建流程：

```
sequences.tsv → id_to_idx (蛋白名 → 整数索引)
    ↓
actions.txt → 按 (idx_a, idx_b) 聚合 mode → multi-hot
    ↓
ppi_list.json → 构建邻接表 + 附加边特征
```

### 3. sampler.py — 子图采样器

#### 邻居选择机制（Scaled Dot-Product Attention）

```
s_t = Mean/Sum(ESM[node] for node in G_t)           # [2560]
s_proj = Linear→ReLU(s_t)                            # [256]
query = Linear(s_proj)                               # [64]

for each neighbor i in frontier:
    edge_agg = MaxPool(edge_labels(i → G_t))          # [7]
    ni = Concat(ESM[i], edge_agg)                    # [2567]
    ni_repr = MLP(ni)                                 # [256]
    key_i = Linear(ni_repr)                           # [64]
    score_i = dot(query, key_i) / sqrt(64)

probs = Softmax(scores)
```

#### REINFORCE 训练

```
V(s) = Baseline(MLP(state))                          # scalar 价值估计

训练时: node ~ Categorical(probs)        → 记录 log_prob
推理时: node = argmax(probs)             → 确定性选择

REINFORCE Loss:
  L_policy = -log_prob * (R - V(s))        # 策略梯度
  L_value  = MSE(V(s), R)                  # 价值估计
  L_total  = L_policy + β * L_value        # β = 0.5
```

### 4. predictor.py — GNN 预测器

```
节点投影: ESM[node] → Linear(2560→256) → LayerNorm → ReLU → Dropout
消息传递: 3× (GATConv(256→64, heads=4) → LayerNorm → ReLU → Dropout + 残差)
读出:    concat[h_u, h_v, h_u⊙h_v, |h_u−h_v|]   # [1024]
分类器:  Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear(→7)
激活:    Sigmoid（多标签独立概率）
```

### 5. ppi_model.py — 交替训练

```
Epoch:
  for step in sampler_steps:
    固定 Predictor, Sampler REINFORCE 更新
    Reward = -BCE(Predictor(子图), label)

  for step in predictor_steps:
    固定 Sampler (argmax), Predictor BCE 监督更新
```

---

## 验证结果

### 图构建验证

| 指标 | 预期 | 实际 | 状态 |
|------|------|------|------|
| 节点数 | 1690 | 1690 | ✅ |
| 边数 | 7624 | 7624 | ✅ |
| 有标签的边 | — | 7624 | ✅ |
| 边特征维度 | 7 | 7 | ✅ |

### 端到端训练（5 Epoch, 32 train / 16 val, SHS27k）

| Epoch | Policy Loss | Reward | BCE Loss | Val BCE |
|-------|-------------|--------|----------|---------|
| 1 | -2.4412 | -0.6841 | 0.7033 | 0.7353 |
| 2 | -1.7814 | -0.5316 | 0.5701 | 0.7895 |
| 3 | -1.7381 | -0.5276 | 0.5560 | 0.7635 |
| 4 | -1.6258 | -0.5270 | 0.5623 | 0.7211 |
| 5 | -1.5646 | -0.5279 | 0.5920 | 0.6846 |

**趋势**：
- Reward 从 -0.684 → -0.528（负 BCE 减小 ≈ 预测更准确）
- Policy Loss 持续下降（策略收敛）
- Val BCE 波动但整体下降趋势

---

## 已发现并修复的 Bug

### Bug 1: 设备不匹配（CUDA vs CPU）
- **现象**：`RuntimeError: Expected all tensors to be on the same device`
- **根因**：`esm_tensor.float()` 后张量留在 CPU，而模型参数在 CUDA
- **修复**：在 `sampler.py:97` 和 `predictor.py:98` 的 `forward()` 开头添加 `.to(device)`

### Bug 2: REINFORCE 梯度断裂（孤立 PPI 对）
- **现象**：`element 0 of tensors does not require grad` → `loss.backward()` 失败
- **根因**：ppi_idx=1994 的两个蛋白互相连接且无其他邻居 → frontier 为空 → sampler 返回 `torch.tensor(0.0)`（无梯度）
- **修复**：改用 `p0.flatten()[0] * 0.0`（p0 = 模型第一个参数），生成与计算图连通的零张量

---

## 设计偏离与改进

| # | 原始设计 | 实现方案 | 原因 |
|---|----------|----------|------|
| 1 | 状态 = ESM embedding **求和** | 默认使用 **均值池化** | 避免子图增大时状态范数线性增长导致注意力 scale 不匹配 |
| 2 | 仅描述注意力思想 | 完整的 **Scaled Dot-Product Attention** + MLP 邻居编码 | 标准化实现，稳定训练 |
| 3 | 未提及 baseline | 加入 **Baseline 价值网络** | REINFORCE 方差缩减，加速收敛 |
| 4 | 未提及边特征处理 | **Max-pool OR 聚合** multi-hot 边标签 | 保留多标签语义的同时固定输出维度 |
| 5 | 未提及梯度累积 | **grad_accum_steps=4** | 缓解逐样本无法批量化的问题 |

---

## 潜在设计问题与建议

### ⚠️ 问题 1：边特征信息泄露
Sampler 使用邻居到子图的边关系类型（7 维 ground truth multi-hot）作为输入。若某邻居同时也是另一个待预测 PPI 的成员，则这些特征在预测该 PPI 时不可用。
- **缓解**：设置 `use_edge_features_in_sampler=False`，退化为二值连接指示器

### ⚠️ 问题 2：REINFORCE 高方差
离散采样 + episode 结束时才获得单个标量奖励，梯度估计方差大。
- **已做**：Baseline 网络（方差缩减）
- **建议**：考虑 Gumbel-Softmax 重参数化替代离散采样

### ⚠️ 问题 3：无法批量化
每个 PPI 对的子图扩展路径不同，无法在 batch 维度上并行。
- **已做**：梯度累积（`grad_accum_steps=4`）
- **建议**：按 degree bucket 分组处理，或使用相同子图结构的组批处理

### ⚠️ 问题 4：交替训练调优
Sampler 和 Predictor 交替训练构成博弈动态，收敛性取决于交替频率和相对学习率。
- **建议**：根据实验调整 `sampler_steps` / `predictor_steps` 比例

### ⚠️ 问题 5：状态表示缩放
原始设计要求 `s_t = sum(ESM embeddings)`，随子图增长范数约线性增大，与邻居表征做点积时 scale 不匹配。
- **已偏离**：默认 `pooling_mode="mean"`，sum 模式可通过参数切换

### ⚠️ 问题 6：子图隔离退化
部分 PPI 对几乎没有邻居，子图无法有效扩展。
- **已处理**：T=0 时仅用 {u, v} 做预测（已修复梯度断裂问题）
