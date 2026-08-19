# Dataset: Protein-Protein Interaction (PPI)

## 概述

本文件夹包含 **Protein-Protein Interaction (PPI)** 相关数据文件

数据用于训练/评估 PPI 预测模型，该模型使用 **ESM-2 3B** 编码蛋白质序列。

---

## 三个数据集总览

| 属性 | SHS27k | SHS148k | STRING |
|------|--------|---------|--------|
| 蛋白质数 | 1,690 | 5,189 | 15,335 |
| PPI 对数 | 7,624 | 44,488 | 593,397 |
| ESM-2 Tensor 形状 | `[1690, 2560]` | `[5189, 2560]` | `[15335, 2560]` |
| Tensor dtype | bfloat16 | bfloat16 | bfloat16 |
| Split 方式 | bfs / dfs / random | dfs / random | dfs |

---

## 文件清单与说明

### 1. 预计算蛋白质嵌入（Tensor）

| 文件 | 形状 | 大小 |
|------|------|------|
| [SHS27k_tensor.pt](SHS27k_tensor.pt) | `[1690, 2560]` | ~8.7 MB |
| [SHS148k_tensor.pt](SHS148k_tensor.pt) | `[5189, 2560]` | ~26.6 MB |
| [STRING_tensor.pt](STRING_tensor.pt) | `[15335, 2560]` | ~78.5 MB |

- 使用 `facebook/esm2_t36_3B_UR50D` 对每个蛋白质的氨基酸序列编码
- **Average Pooling**：对残基级别的 last hidden state 做平均池化，得到蛋白质级别的嵌入向量
- `tensor[i]` = Protein Index `i` 的嵌入向量（dtype: `torch.bfloat16`）

### 2. PPI 索引映射

| 文件 | PPI 对数 | 大小 |
|------|----------|------|
| [SHS27k_ppi_list.json](SHS27k_ppi_list.json) | 7,624 | ~90 KB |
| [SHS148k_ppi_list.json](SHS148k_ppi_list.json) | 44,488 | ~570 KB |
| [STRING_ppi_list.json](STRING_ppi_list.json) | 593,397 | ~8.1 MB |

- 格式：`[[a, b], [a, b], ...]`，每个元素 `[a, b]` 中的整数是 **Protein Index**（0-based）
- 数组长度 = PPI 对总数，数组下标 = **PPI Index**

### 3. 蛋白质序列字典

| 文件 | 蛋白数 | 大小 |
|------|--------|------|
| [protein.SHS27k.sequences.dictionary.tsv](protein.SHS27k.sequences.dictionary.tsv) | 1,690 | ~1.0 MB |
| [protein.SHS148k.sequences.dictionary.tsv](protein.SHS148k.sequences.dictionary.tsv) | 5,189 | ~3.2 MB |
| [protein.STRING.sequences.dictionary.tsv](protein.STRING.sequences.dictionary.tsv) | 15,335 | ~9.6 MB |

- TSV 格式：`id\tseq`（有 header）
- 第 `i` 行数据（0-based，跳过 header）= **Protein Index `i`**
- `id` 列格式：`{taxonomy_id}.{ENSEMBL_ID}`（如 `9606.ENSP00000000233`）
- `seq` 列为完整氨基酸序列

### 4. PPI 关系标注（Actions）

| 文件 | 大小 |
|------|------|
| [protein.actions.SHS27k.txt](protein.actions.SHS27k.txt) | ~4.0 MB |
| [protein.actions.SHS148k.txt](protein.actions.SHS148k.txt) | ~23.0 MB |
| [protein.actions.STRING.txt](protein.actions.STRING.txt) | ~290 MB |

- TSV 格式，列：`item_id_a`, `item_id_b`, `mode`, `action`, `is_directional`, `a_is_acting`, `score`
- **同一个无向蛋白对可能有多行**；即使 `(pair, mode)` 相同也可能重复出现，需先将端点排序规范化，再按无向 pair 对各 `mode` 做 **OR 聚合**
- 蛋白对是无向的（排序后去重）

### 5. 训练/验证/测试集划分（Split）

| 文件 | Split 方式 | Train | Val | Test |
|------|-----------|-------|-----|------|
| [SHS27k_bfs.json](SHS27k_bfs.json) | BFS | 4,562 | 1,524 | 1,538 |
| [SHS27k_dfs.json](SHS27k_dfs.json) | DFS | 4,571 | 1,524 | 1,529 |
| [SHS27k_random.json](SHS27k_random.json) | Random | 4,574 | 1,525 | 1,525 |
| [SHS148k_dfs.json](SHS148k_dfs.json) | DFS | 26,599 | 8,897 | 8,992 |
| [SHS148k_random.json](SHS148k_random.json) | Random | 26,692 | 8,898 | 8,898 |
| [STRING_dfs.json](STRING_dfs.json) | DFS | 355,891 | 118,679 | 118,827 |

- JSON 格式：`{"train_index": [...], "val_index": [...], "test_index": [...]}`
- 数组中的值均为 **PPI Index**（不是 Protein Index）
- **BFS/DFS**：按图结构遍历结果划分边集合，用于评估结构偏移下的泛化能力；当前数据中测试 PPI 的两个端点不会同时出现在训练节点集合中，但可以与训练集共享一个端点，并不表示 train 与 val/test 是互不连通的分量
- **Random**：随机打乱后按 60/20/20 划分

## 三层索引映射链

```
PPI Index (0 ~ N-1)
    ↓ 通过 {name}_ppi_list.json
[protein_idx_a, protein_idx_b]  (0 ~ M-1)
    ↓ 通过 protein.{name}.sequences.dictionary.tsv 的行号
ENSEMBL ID + 氨基酸序列
    ↓ 通过 protein.actions.{name}.txt 查找该蛋白对
Multi-hot Label (7维向量)
```

---

## Label 体系（7 类 PPI 关系）

| 编号 | Label | 含义 |
|------|-------|------|
| 0 | reaction | 反应 |
| 1 | binding | 结合 |
| 2 | ptmod | 翻译后修饰 |
| 3 | activation | 激活 |
| 4 | inhibition | 抑制 |
| 5 | catalysis | 催化 |
| 6 | expression | 表达 |

每个 PPI 对应一个长度为 7 的 multi-hot 向量（`[0/1, ...]`），可同时具有多种关系。

### 各数据集 Label 分布（按 actions.txt 中 mode 出现行数）

| Label | SHS27k | SHS148k | STRING |
|-------|--------|---------|--------|
| reaction | 18,162 | 102,964 | 1,669,750 |
| binding | 16,056 | 93,632 | 1,610,314 |
| catalysis | 11,796 | 67,168 | 998,266 |
| activation | 7,400 | 42,516 | 232,240 |
| inhibition | 5,550 | 34,712 | 147,676 |
| ptmod | 2,872 | 20,154 | 88,424 |
| expression | 1,572 | 7,896 | 28,484 |

表中数值是 actions.txt 中该 `mode` 出现的**总行数**（同一蛋白对、甚至同一 pair+mode 可能有多行）。
训练实际使用的是对每个 PPI 对按 `mode` 做 OR 聚合后的 multi-hot 标签，其各类别计数
小于或等于原始行数（SHS27k 各类别原始行数约为聚合后 PPI 数的 2.2~5.7 倍），
由 `src/ppi_graph.py` 在加载时计算。

---

## 快速使用示例

```python
import json, pandas as pd, csv, ast

# 1. 加载 PPI 索引映射
ppi_list = json.load(open('SHS27k_ppi_list.json'))

# 2. 加载蛋白质字典（行号 = Protein Index）
with open('protein.SHS27k.sequences.dictionary.tsv') as f:
    proteins = list(csv.DictReader(f, delimiter='\t'))

# 3. 加载 split
split = json.load(open('SHS27k_bfs.json'))

# 4. 查询某个训练样本
ppi_idx = split['train_index'][0]
a_idx, b_idx = ppi_list[ppi_idx]

print(f"PPI {ppi_idx}:")
print(f"  Protein A (idx={a_idx}): {proteins[a_idx]['id']}")
print(f"    Seq: {proteins[a_idx]['seq'][:60]}...")
print(f"  Protein B (idx={b_idx}): {proteins[b_idx]['id']}")
print(f"    Seq: {proteins[b_idx]['seq'][:60]}...")

# 5. 查询标签（从 actions.txt）
name_a, name_b = proteins[a_idx]['id'], proteins[b_idx]['id']
df_act = pd.read_csv('protein.actions.SHS27k.txt', sep='\t')
mask = ((df_act['item_id_a'] == name_a) & (df_act['item_id_b'] == name_b)) | \
       ((df_act['item_id_a'] == name_b) & (df_act['item_id_b'] == name_a))
labels = df_act[mask]['mode'].unique().tolist()
print(f"  Labels: {labels}")
```

### 使用预计算 Tensor

```python
import torch

tensor = torch.load('SHS27k_tensor.pt')  # [1690, 2560] bfloat16
# tensor[i] 对应 Protein Index i 的嵌入
emb_a = tensor[a_idx]  # Protein A 的 ESM-2 嵌入
emb_b = tensor[b_idx]  # Protein B 的 ESM-2 嵌入
```

---

## 数据分析：孤立 PPI 与跨 Split 节点分布

### 孤立 PPI 对统计

**定义**：孤立 PPI 对 (u, v) 满足 degree(u) = degree(v) = 1，即两蛋白仅与彼此相互作用。松散孤立：至少一端度为 1。

| 数据集 | 蛋白数 | PPI 总数 | 度为1蛋白 | 孤立PPI(双端度=1) | 松散孤立(一端度=1) |
|--------|:-----:|:--------:|:--------:|:--------------:|:---------------:|
| SHS27k | 1,690 | 7,624 | 409 | **14** | 395 |
| SHS148k | 5,189 | 44,488 | 1,016 | **6** | 1,010 |
| STRING | 15,335 | 593,397 | 1,044 | **0** | 1,044 |

STRING 有 1,044 个度为 1 的蛋白，但它们两两之间没有形成 PPI 对——每个度为 1 的蛋白连接的是图中度较高的蛋白，因此不存在双端度=1 的孤立 PPI。

#### 各 Split 训练集内部孤立 PPI（在训练子图内计算度）

| 数据集 | Split | Train PPIs | 训练子图孤立PPI |
|--------|-------|:----------:|:-----------:|
| SHS27k | bfs | 4,562 | 20 |
| SHS27k | dfs | 4,571 | 20 |
| SHS27k | random | 4,574 | 19 |
| SHS148k | dfs | 26,599 | 9 |
| SHS148k | random | 26,692 | 11 |
| STRING | dfs | 355,891 | 11 |

**BFS/DFS 划分**：当前文件中的全局孤立 PPI 都落在 train；这与测试 PPI 不会两端同时出现在训练节点集合中的可见性统计同时成立，但不能解释为 train 和 val/test 来自互不连通的图分量。Random 划分下孤立 PPI 可能被分配到 val/test，当前文件中确实存在这种情况。

### 测试集 PPI 节点在训练集中的出现情况

分析测试集 PPI 对 (u, v) 的两个蛋白节点是否出现在训练集中：

| 数据集 | Split | Test PPIs | 两节点均在训练 | 仅一个在训练 | 均不在训练 |
|--------|-------|:---------:|:----------:|:--------:|:--------:|
| SHS27k | bfs | 1,538 | 0 (0%) | 1,078 (70.1%) | 460 (29.9%) |
| SHS27k | dfs | 1,529 | 0 (0%) | 1,215 (79.5%) | 314 (20.5%) |
| SHS27k | random | 1,525 | 1,377 (90.3%) | 138 (9.0%) | 10 (0.7%) |
| SHS148k | dfs | 8,992 | 0 (0%) | 7,739 (86.1%) | 1,253 (13.9%) |
| SHS148k | random | 8,898 | 8,598 (96.6%) | 294 (3.3%) | 6 (0.1%) |
| STRING | dfs | 118,827 | 0 (0%) | 103,710 (87.3%) | 15,117 (12.7%) |

**关键结论**：
- **BFS/DFS**："两节点均在训练集"恒为 0（当前划分的节点可见性约束，不等同于互不连通分量）。70-87% 的测试 PPI 有一个节点在训练集中出现过，12-30% 的两个节点都从未出现。
- **Random**：90-97% 的测试 PPI 两节点均在训练集中存在，极少数（0.1-0.7%）均未出现（对应低度蛋白的所有相互作用恰好分入 val/test）。

## 参考文献

- ESM-2 模型：`facebook/esm2_t36_3B_UR50D`
- 数据来源：STRING 数据库 (SHS27k / SHS148k / STRING)
