# PPI 预测项目 — 开发报告

> 本文件是**持续维护**的项目开发报告，记录项目背景、代码设计、关键决策与验证结果。
> 每完成一轮开发/验证后更新「开发日志」与「当前状态」；改动设计时更新对应小节并说明原因。
> 数据集本身的详细格式说明见 [dataset/CLAUDE.md](dataset/CLAUDE.md)（数据格式由数据侧定义，本文件不复述全部细节，只记录与代码相关的关键约束）。

---

## 1. 项目目标

使用预计算的 **ESM-2 3B** 蛋白质嵌入，预测蛋白质-蛋白质相互作用（PPI）的多标签（7 类关系）。

- 输入：蛋白质对 `(u, v)` 的 ESM-2 嵌入（`[M, 2560]`, bfloat16）
- 输出：7 维 multi-hot label（见 [dataset/CLAUDE.md](dataset/CLAUDE.md) 的 Label 体系）
- 训练环境：PyTorch，后续在 **CUDA** 上训练

## 2. 数据集（概要）

| 属性 | SHS27k | SHS148k | STRING |
|------|--------|---------|--------|
| 蛋白质数 | 1,690 | 5,189 | 15,335 |
| PPI 对数 | 7,624 | 44,488 | 593,397 |
| 嵌入张量 | `[1690, 2560]` bf16 | `[5189, 2560]` bf16 | `[15335, 2560]` bf16 |
| 可用 Split | bfs / dfs / random | dfs / random | dfs |

**三层索引映射链**（代码依赖的关键）：

```
PPI Index (0..N-1)
    → {name}_ppi_list.json        : [protein_a, protein_b]
Protein Index (0..M-1)
    → protein.{name}.sequences.dictionary.tsv  (行号 = Protein Index, id/seq)
    → {name}_tensor.pt            : tensor[i] = 嵌入向量
Label (7 维 multi-hot)
    → protein.actions.{name}.txt  : (item_id_a, item_id_b, mode, ...) OR 聚合
Split
    → {name}_{split}.json         : {"train_index": [...], "val_index": [...], "test_index": [...]}
```

**7 类 Label**（代码中 `PPIGraph.LABELS`，索引即 multi-hot 位置）：
`0 reaction, 1 binding, 2 ptmod, 3 activation, 4 inhibition, 5 catalysis, 6 expression`

## 3. 代码结构

```
src/
  __init__.py
  ppi_graph.py    # PPIGraph 类：加载数据集 + 逐 split 建图 + DataLoader
```

## 4. PPIGraph 类设计

文件：[src/ppi_graph.py](src/ppi_graph.py)

### 4.1 构造

```python
PPIGraph(name='SHS27k', split='bfs', root='dataset', device='cpu', cache_dir=None)
```

- `name` ∈ {SHS27k, SHS148k, STRING}；`split` 按 `AVAILABLE_SPLITS` 校验（如 STRING 仅 dfs）。
- `device`：特征张量与标签存放设备（训练时传 `'cuda'`）。
- `cache_dir`：可选，缓存构建好的 `[N, 7]` 标签张量为 `{name}_ppi_labels.pt`，避免重复解析大 actions.txt（STRING 首次约 13s，缓存后约 2.8s）。

构造时完成全部加载：
- `self.tensor`：`[M, 2560]` bf16 嵌入
- `self.ppi_list`：PPI Index → [protein_a, protein_b]
- `self.proteins` / `self.protein_id2idx`：蛋白字典（行号 = Protein Index）
- `self.split_index`：train/val/test 的 PPI Index
- `self.ppi_labels`：`[N, 7]` float32 multi-hot（对 actions.txt 做无向对 mode OR 聚合）
- `self.ppi`：`[N, 2]` LongTensor，`ppi[i]` = PPI i 的蛋白索引

### 4.2 逐 split 建图

```python
build_graph(split_name='train', remap_nodes=True, undirected=True) -> dict
```

返回：`edge_index [2, E]`、`edge_label [E, 7]`、`node_index [N]`、`node_feat [N, 2560]`。

- `remap_nodes=True`（默认）：节点重映射为局部连续 id，`node_feat` 子集化，便于 GNN 直接 `node_feat[node]`。
- `undirected=True`（默认）：**补全反向边**——`ppi_list` 中每个无向 PPI 对只出现一次且方向任意，故每个对生成 `(u,v)` 与 `(v,u)` 两条边，`edge_label` 对半重复对齐。`False` 时保持单向。

### 4.3 DataLoader

```python
get_dataloader(split_name='train', batch_size=32, shuffle=None, num_workers=0,
               drop_last=False, pin_memory=False, collate_fn=None, **kwargs) -> DataLoader
```

- 每个样本 `(u, v, label)`：`u, v` 为 `[B]` 蛋白索引 LongTensor，`label` 为 `[B, 7]` float32。
- `shuffle` 默认 train=True，val/test=False。
- 数据集类：`_PPIDataset`（见 [src/ppi_graph.py](src/ppi_graph.py)）。

## 5. 关键决策与依据（开发报告核心）

| 决策 | 理由 |
|------|------|
| 标签用 actions.txt 流式解析，而非读 ml.csv | ml.csv 含完整序列（STRING 达 1.5GB）；actions.txt 只需 3 列、可逐行处理，内存友好 |
| 无向对 mode **OR 聚合**，key 排序归一化 | 同一蛋白对在 actions.txt 有多行（不同 mode/is_directional/a_is_acting），对方向散乱存放 |
| **补全反向边**构建无向图 | **已验证**：ppi_list 中每个无向对恰好出现一次、方向任意（如 SHS27k 有 2742/7624 条以 a>b 列出），PPI 相互作用本身无向，消息传递需双向 |
| Dataloader **不改**（每对输出一次 (u,v)） | 模型侧保证对 `(u,v)`/`(v,u)` 输出对称（用户明确要求） |
| 标签缓存 | STRING actions.txt 290MB，解析成本高；缓存 `[N,7]` 张量避免重复 |
| 嵌入保持 bf16 | 数据集原生 dtype，训练时由模型按需转换 |

## 6. 已验证结果

- 7 种 mode 与 `LABELS` 完全一致；所有数据集 `ppi_list` 的每个 PPI 对都能在 actions.txt 找到标签（SHS27k 缺标 0 对）。
- 方向性分析：ppi_list 无重复无向对、无自环；actions.txt 中 `a > b` 行约占一半（SHS27k 31704/63408），确认无向对方向散乱。
- 补全反向边后：SHS27k train 边数 4562 → 9124，逐边反向存在、前后半标签一致；`undirected=False` 时边数 = PPI 对数。
- 全流程 smoke test（SHS27k/STRING 的加载、三 split 建图、DataLoader、缓存、非法 split 校验）通过。

## 7. 开发日志

- **R1**：完成 `PPIGraph` 类（加载 + 逐 split 建图 + DataLoader）。
- **R2**：验证「ppi_list 只列单向 PPI 对」断言成立；`build_graph` 新增 `undirected=True` 补全反向边；修复拼接时 `u` 被提前重赋值导致 `v = cat([v, u])` 长度错乱（3E）的 bug。

## 8. 后续 / 待办

- 深度学习模型设计（嵌入特征 → 边分类 / GNN 消息传递），确保 `(u,v)`/`(v,u)` 输出对称。
- CUDA 训练脚本（`get_dataloader` 已支持 `pin_memory` 与 `device='cuda'`）。
