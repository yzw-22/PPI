# SHS27k BFS/DFS Sampler 消融实验

## 1. 实验设置

本报告记录当前实现下 SHS27k 的 BFS 和 DFS 两套 split 消融实验。两套实验均使用相同配置：

- seed `42`，训练 `10` 个 epoch，设备 `cuda`；
- batch size `32`，eval batch size `64`，hidden size `256`；
- `fixed_num=1`，learned Sampler 的 `max_steps=10`；
- GNN layers `2`，attention heads `4`，dropout `0.1`；
- Sampler learning rate `1e-4`，Predictor learning rate `1e-3`，`reinforce_gamma=1.0`。

测试集只在验证 Macro-AUC 最佳 checkpoint 上评估。BS/ES/NS 的定义是测试 PPI 两端相对于训练节点集合的可见性；本次 BFS 和 DFS 的 BS 分组均为空。

### Sampler 模式

| 模式 | 行为 |
|---|---|
| `learned` | 当前 REINFORCE Sampler；从 `G0` 开始最多执行 `max_steps` 次决策，含 STOP action。|
| `target_only` | 只保留目标节点 `u,v`。|
| `target_proxy` | 保留目标节点及必要 proxy，不增加上下文节点。|
| `random_1hop10` | 从 `{u,v,proxy}` 的安全一跳邻居并集随机选择最多 10 个上下文节点，seed 固定为 42。|
| `random_iterative10` | 复用 learned 的 `G0`，从当前 frontier 每步随机增加一个节点，最多增加 10 个节点，seed 固定为 42；不执行 REINFORCE。|

所有模式都移除目标边的两个方向，并限制节点、proxy 和边属于当前 split。固定基线不更新 Sampler，只训练 Predictor。

## 2. 复现实验命令

### BFS

```bash
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode learned --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_learned.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode target_only --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_target_only.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode target_proxy --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_target_proxy.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode random_1hop10 --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_random_1hop10.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode random_iterative10 --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_random_iterative10.json
```

### DFS

```bash
python -m src.train_shs27k --dataset SHS27k --split dfs --sampler-mode learned --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_dfs_learned.json
python -m src.train_shs27k --dataset SHS27k --split dfs --sampler-mode target_only --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_dfs_target_only.json
python -m src.train_shs27k --dataset SHS27k --split dfs --sampler-mode target_proxy --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_dfs_target_proxy.json
python -m src.train_shs27k --dataset SHS27k --split dfs --sampler-mode random_1hop10 --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_dfs_random_1hop10.json
python -m src.train_shs27k --dataset SHS27k --split dfs --sampler-mode random_iterative10 --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_dfs_random_iterative10.json
```

仓库中的 `run.sh` 保持默认执行 BFS 五组实验；DFS 命令如上单独执行，避免一次运行十组长实验。

## 3. BFS 结果

BFS 测试集为 1538 个 PPI（ES 1078，NS 460）。表中测试指标来自最佳验证 checkpoint。

| 模式 | 时间(s) | 最佳 epoch | 最佳验证 Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 1430.0 | 6 | 0.7332 | 0.7214 | **0.7705** | 0.4947 | **0.5882** |
| target_only | 31.5 | 9 | 0.7389 | 0.7422 | 0.7635 | 0.5181 | 0.5331 |
| target_proxy | 38.3 | 7 | 0.7413 | 0.7367 | 0.7632 | 0.5135 | 0.5648 |
| random_1hop10 | 42.8 | 10 | **0.7565** | 0.7457 | 0.7604 | **0.5325** | 0.5691 |
| random_iterative10 | 149.0 | 9 | 0.7526 | **0.7466** | 0.7693 | 0.5215 | 0.5406 |

### BFS 每 epoch（Predictor loss / Val Macro-AUC / Val Macro-F1）

| epoch | learned | target_only | target_proxy | random_1hop10 | random_iterative10 |
|---:|---|---|---|---|---|
| 1 | 0.4788 / 0.7191 / 0.3272 | 0.4878 / 0.7020 / 0.4221 | 0.4890 / 0.6777 / 0.4211 | 0.4859 / 0.7032 / 0.3902 | 0.4903 / 0.7070 / 0.3446 |
| 2 | 0.3782 / 0.7317 / 0.3496 | 0.3718 / 0.7209 / 0.4025 | 0.3735 / 0.7303 / 0.4261 | 0.3853 / 0.7438 / 0.3831 | 0.4069 / 0.7318 / 0.3838 |
| 3 | 0.3295 / 0.7179 / 0.3442 | 0.3240 / 0.7379 / 0.3619 | 0.3215 / 0.7270 / 0.2979 | 0.3330 / 0.7448 / 0.3257 | 0.3472 / 0.7307 / 0.3555 |
| 4 | 0.2880 / 0.7027 / 0.3569 | 0.2879 / 0.7116 / 0.3751 | 0.2859 / 0.7234 / 0.3922 | 0.2886 / 0.7346 / 0.4127 | 0.3014 / 0.7312 / 0.4058 |
| 5 | 0.2584 / 0.7331 / 0.4514 | 0.2555 / 0.7367 / 0.5004 | 0.2538 / 0.7312 / 0.4668 | 0.2536 / 0.7371 / 0.4624 | 0.2636 / 0.7355 / 0.4909 |
| 6 | 0.2307 / 0.7332 / 0.5080 | 0.2333 / 0.7205 / 0.4590 | 0.2309 / 0.7224 / 0.5026 | 0.2287 / 0.7438 / 0.4773 | 0.2349 / 0.7292 / 0.4632 |
| 7 | 0.2090 / 0.7179 / 0.4650 | 0.2139 / 0.7182 / 0.4769 | 0.2096 / 0.7413 / 0.5120 | 0.2082 / 0.7371 / 0.4709 | 0.2172 / 0.7304 / 0.4690 |
| 8 | 0.1882 / 0.7188 / 0.4551 | 0.1936 / 0.7364 / 0.5134 | 0.1958 / 0.7267 / 0.5201 | 0.1864 / 0.7339 / 0.5322 | 0.1964 / 0.7448 / 0.5094 |
| 9 | 0.1748 / 0.7045 / 0.4430 | 0.1861 / 0.7389 / 0.5194 | 0.1804 / 0.7312 / 0.5232 | 0.1799 / 0.7498 / 0.5140 | 0.1789 / 0.7526 / 0.5242 |
| 10 | 0.1688 / 0.7168 / 0.4400 | 0.1724 / 0.7248 / 0.4986 | 0.1722 / 0.7294 / 0.5022 | 0.1590 / 0.7565 / 0.5561 | 0.1642 / 0.7344 / 0.5063 |

### BFS 可见性指标

格式：`ROC-AUC Macro / ROC-AUC Micro / F1 Macro / F1 Micro`。

| 模式 | ES（1078） | NS（460） |
|---|---|---|
| learned | 0.7324 / **0.7741** / 0.4990 / **0.5886** | 0.6936 / **0.7623** / **0.4828** / **0.5871** |
| target_only | 0.7568 / 0.7807 / 0.5409 / 0.5538 | 0.7020 / 0.7201 / 0.4544 / 0.4815 |
| target_proxy | 0.7590 / **0.7852** / 0.5454 / **0.5919** | 0.6830 / 0.7064 / 0.4307 / 0.4968 |
| random_1hop10 | 0.7518 / 0.7696 / **0.5537** / 0.5866 | **0.7324** / 0.7376 / 0.4765 / 0.5252 |
| random_iterative10 | **0.7587** / 0.7809 / 0.5449 / 0.5618 | 0.7168 / 0.7399 / 0.4574 / 0.4879 |

### BFS 子图诊断

| 模式 | 平均 steps | 平均最终节点 | 平均上下文节点 | 平均真实边 |
|---|---:|---:|---:|---:|
| learned | 9.98 | 14.03 | 11.94 | 15.80 |
| target_only | 0 | 2.00 | 0 | 0 |
| target_proxy | 0 | 2.09 | 0 | 约 0 |
| random_1hop10 | 0 | 11.66 | 9.57 | 15.98 |
| random_iterative10 | 9.99 | 14.05 | 11.95 | 16.82 |

## 4. DFS 结果

DFS 测试集为 1529 个 PPI（ES 1215，NS 314）。

| 模式 | 时间(s) | 最佳 epoch | 最佳验证 Macro-AUC | Test Macro-AUC | Test Micro-AUC | Test Macro-F1 | Test Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 1474.9 | 9 | 0.8082 | 0.8149 | 0.8280 | 0.5569 | 0.6121 |
| target_only | 35.9 | 10 | 0.8265 | **0.8212** | **0.8430** | 0.5966 | **0.6608** |
| target_proxy | 38.7 | 10 | **0.8283** | 0.8184 | 0.8336 | 0.5930 | 0.6298 |
| random_1hop10 | 42.4 | 10 | 0.8130 | 0.8169 | 0.8379 | 0.6034 | 0.6329 |
| random_iterative10 | 145.9 | 9 | 0.8123 | 0.8029 | 0.8282 | 0.5824 | 0.6259 |

### DFS 每 epoch（Predictor loss / Val Macro-AUC / Val Macro-F1）

| epoch | learned | target_only | target_proxy | random_1hop10 | random_iterative10 |
|---:|---|---|---|---|---|
| 1 | 0.5176 / 0.7350 / 0.3913 | 0.5183 / 0.7440 / 0.5269 | 0.5151 / 0.7297 / 0.4252 | 0.5144 / 0.7507 / 0.3986 | 0.5177 / 0.7306 / 0.3831 |
| 2 | 0.4242 / 0.7589 / 0.4964 | 0.4093 / 0.7781 / 0.5114 | 0.4091 / 0.7756 / 0.4819 | 0.4296 / 0.7577 / 0.4023 | 0.4394 / 0.7419 / 0.3811 |
| 3 | 0.3730 / 0.7636 / 0.4677 | 0.3549 / 0.7949 / 0.5450 | 0.3545 / 0.7924 / 0.5457 | 0.3786 / 0.7748 / 0.5064 | 0.3877 / 0.7681 / 0.5064 |
| 4 | 0.3225 / 0.7779 / 0.5336 | 0.3202 / 0.7877 / 0.5246 | 0.3195 / 0.7922 / 0.5315 | 0.3306 / 0.7840 / 0.5310 | 0.3409 / 0.7846 / 0.5321 |
| 5 | 0.2798 / 0.7966 / 0.5097 | 0.2905 / 0.8118 / 0.4930 | 0.2900 / 0.8009 / 0.4793 | 0.2890 / 0.8038 / 0.5172 | 0.3019 / 0.7931 / 0.5135 |
| 6 | 0.2570 / 0.7937 / 0.5622 | 0.2610 / 0.8133 / 0.5780 | 0.2635 / 0.8132 / 0.6021 | 0.2629 / 0.8004 / 0.5887 | 0.2729 / 0.8065 / 0.5722 |
| 7 | 0.2323 / 0.8004 / 0.5801 | 0.2440 / 0.8129 / 0.6005 | 0.2435 / 0.8131 / 0.5841 | 0.2424 / 0.7962 / 0.5618 | 0.2468 / 0.7979 / 0.5822 |
| 8 | 0.2146 / 0.7830 / 0.5301 | 0.2266 / 0.8102 / 0.5682 | 0.2241 / 0.8156 / 0.5799 | 0.2182 / 0.8030 / 0.5729 | 0.2248 / 0.7926 / 0.5484 |
| 9 | 0.1997 / 0.8082 / 0.5608 | 0.2089 / 0.8130 / 0.5855 | 0.2046 / 0.8255 / 0.6155 | 0.1983 / 0.8101 / 0.6142 | 0.2064 / 0.8123 / 0.6086 |
| 10 | 0.1834 / 0.7980 / 0.5702 | 0.1974 / 0.8265 / 0.6247 | 0.1943 / 0.8283 / 0.6146 | 0.1852 / 0.8130 / 0.5930 | 0.1857 / 0.8115 / 0.5853 |

### DFS 可见性指标

| 模式 | ES（1215） | NS（314） |
|---|---|---|
| learned | 0.8227 / 0.8358 / 0.5757 / 0.6306 | 0.7795 / 0.7920 / 0.4693 / 0.5273 |
| target_only | **0.8378 / 0.8588 / 0.6214 / 0.6866** | 0.7432 / 0.7657 / 0.4809 / 0.5467 |
| target_proxy | 0.8355 / 0.8491 / 0.6193 / 0.6559 | 0.7406 / 0.7609 / 0.4768 / 0.5119 |
| random_1hop10 | 0.8330 / 0.8522 / **0.6263 / 0.6600** | 0.7416 / 0.7715 / **0.4957** / 0.5115 |
| random_iterative10 | 0.8154 / 0.8391 / 0.6060 / 0.6493 | **0.7476 / 0.7765** / 0.4735 / 0.5194 |

### DFS 子图诊断

| 模式 | 平均 steps | 平均最终节点 | 平均上下文节点 | 平均真实边 |
|---|---:|---:|---:|---:|
| learned | 9.98 | 14.04 | 11.95 | 16.10 |
| target_only | 0 | 2.00 | 0 | 0 |
| target_proxy | 0 | 2.09 | 0 | 0 |
| random_1hop10 | 0 | 11.68 | 9.59 | 16.52 |
| random_iterative10 | 9.99 | 14.06 | 11.97 | 17.13 |

## 5. learned 相对基线的差值

以下均为测试集指标的 `learned - baseline`。

### BFS

| 对比 | Macro-AUC | Micro-AUC | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|
| target_only | -0.0208 | +0.0070 | -0.0234 | +0.0551 |
| target_proxy | -0.0153 | +0.0073 | -0.0188 | +0.0233 |
| random_1hop10 | -0.0242 | +0.0102 | -0.0378 | +0.0190 |
| random_iterative10 | -0.0252 | +0.0013 | -0.0268 | +0.0475 |

### DFS

| 对比 | Macro-AUC | Micro-AUC | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|
| target_only | -0.0063 | -0.0150 | -0.0396 | -0.0487 |
| target_proxy | -0.0035 | -0.0056 | -0.0361 | -0.0177 |
| random_1hop10 | -0.0020 | -0.0098 | -0.0465 | -0.0208 |
| random_iterative10 | +0.0120 | -0.0002 | -0.0255 | -0.0138 |

## 6. 结论与限制

1. learned 在两个 split 上都没有稳定提升 Macro-AUC 或 Macro-F1。DFS 中 `target_only` 的测试 Macro/Micro-AUC 和 Micro-F1 最好；BFS 中随机方法的 Macro 指标最好。
2. BFS 中 learned 的 Micro-AUC 和 Micro-F1 最高，但 Macro 指标较低，不能据此断言 Sampler 提升了整体类别均衡性能。
3. learned 与 `random_iterative10` 的子图规模几乎相同；后者效果不优于小规模基线，说明当前性能差异不是单纯由节点预算造成。
4. learned 的训练代价约为固定基线的 10--45 倍，正式扩大 seed 数量前应优先优化 G0 缓存、图构造、重复 Predictor forward 和 Python frontier 操作。
5. 这些结果来自单个 seed、每个 split 一次运行，只能作为当前实现的诊断性证据；要做稳定结论仍需多 seed 配对实验。

原始 JSON 位于 `/tmp/ppi_shs27k_{bfs,dfs}_*.json`，不属于仓库数据文件。
