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
  sampler.py      # 单样本子图采样器：REINFORCE 轨迹与虚拟代理
  predictor.py    # GAT PPI 多标签预测器
  trainer.py      # Sampler / Predictor 交替训练
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
- `self.proteins`：蛋白字典（行号 = Protein Index）
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
- `pin_memory=True` 仅适用于 `device='cpu'`；当数据已在 CUDA 上时不应再 pin，且 CUDA 数据集要求 `num_workers=0`。
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
- **R3**：审查 `ppi_graph.py`：删除错误的 `pin_memory` 重写逻辑；CUDA 数据集禁止 `pin_memory` 和多进程 worker；使用安全的 `weights_only=True` 加载张量。
- **R4**：删除未被实现消费的 `protein_id2idx` 冗余预计算，并同步接口说明。
- **R5**：按 `src/CLAUDE.md` 完成单样本 `SubgraphSampler`、`PPIPredictor` 与 `AlternatingTrainer`；采样仅使用当前 split 拓扑，并移除目标边防止标签泄露。
- **R6**：移除额外的 split 完整性扫描及 `get_dataloader()` 中重复的 `split_name` 检查，默认数据文件按数据集约定有效。
- **R7**：增加批量 Predictor/GAT 训练与 SHS27k/bfs 实验入口，完成 CUDA 全量 10 epoch 训练。

## 8. 后续 / 待办

- 扩展模型训练与评估脚本，当前 `PPIPredictor` 已支持嵌入特征 → GAT 边分类，并确保 `(u,v)`/`(v,u)` 输出对称。
- CUDA 训练脚本（建议图数据放在 CUDA 上并使用 `num_workers=0`、`pin_memory=False`；若需 pin memory，应让图保持在 CPU 上并在训练循环中搬运 batch）。
- 更高效的批量采样决策；当前实现已支持批量 Predictor，但采样动作仍逐样本生成。

## 9. 本轮审查结论

- 已修复：原实现把 `pin_memory` 在 CUDA 上强制打开、在 CPU 上强制关闭，方向相反；CUDA tensor 不能被 pin，CPU batch 则因此失去异步搬运收益。
- 已调整：移除不会参与建图结果的 split 完整性扫描；split 文件按数据集约定直接读取，具体索引错误由后续张量索引自然暴露。
- 仍需注意：标签缓存文件名只包含数据集名，不包含 `root` 或数据版本；多个不同数据根目录共用同一个 `cache_dir` 时可能误用旧缓存。当前接口仍假定调用方为每套数据使用独立缓存目录。

## 10. SHS27k/bfs 全量训练结果

配置：CUDA、seed=42、batch size=32、hidden=256、GAT 2 层/4 heads、最大采样步数 3、Sampler lr=1e-4、Predictor lr=1e-3。训练集 4,562 对，测试集 1,538 对，耗时约 530 秒。F1 使用 sigmoid 阈值 0.5。

| Epoch | Sampler loss | Predictor loss | Mean reward | Test ROC-AUC macro | Test ROC-AUC micro | Test F1 macro | Test F1 micro |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.005490 | 0.480742 | 0.006716 | 0.709452 | 0.747444 | 0.444534 | 0.512271 |
| 2 | -0.003628 | 0.374455 | 0.043777 | 0.712051 | 0.743736 | 0.380833 | 0.440964 |
| 3 | -0.001164 | 0.310331 | 0.054743 | 0.728985 | 0.743354 | 0.420607 | 0.490469 |
| 4 | -0.000344 | 0.273286 | 0.059335 | 0.732441 | 0.770116 | 0.484489 | 0.542383 |
| 5 | 0.002193 | 0.239935 | 0.067391 | 0.731386 | 0.757221 | 0.496450 | 0.524898 |
| 6 | 0.002004 | 0.212568 | 0.069630 | 0.721438 | 0.737533 | 0.437038 | 0.463569 |
| 7 | 0.000728 | 0.188540 | 0.068576 | 0.756130 | 0.786363 | 0.537962 | 0.593515 |
| 8 | 0.001988 | 0.172383 | 0.069906 | 0.749133 | 0.779100 | 0.543118 | 0.588333 |
| 9 | 0.001842 | 0.159224 | 0.067951 | 0.755753 | 0.787892 | 0.535114 | 0.585588 |
| 10 | 0.002193 | 0.140889 | 0.069434 | 0.746808 | 0.780301 | 0.507484 | 0.564287 |

最佳测试 Macro ROC-AUC 为 epoch 7 的 0.756130；最佳 Micro ROC-AUC 为 epoch 9 的 0.787892；最佳 Macro-F1 为 epoch 8 的 0.543118；最佳 Micro-F1 为 epoch 7 的 0.593515。完整结果保存于 `shs27k_bfs_10epoch.json`。
