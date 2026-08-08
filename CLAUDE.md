# PPI 项目开发报告

本文件记录当前实现与已验证的设计约束。数据格式以 [dataset/CLAUDE.md](dataset/CLAUDE.md) 为准，模型细节见 [src/CLAUDE.md](src/CLAUDE.md)。

## 项目目标

使用预计算的 ESM-2 3B 蛋白质嵌入预测 7 类 PPI 多标签关系。核心代码位于 `src/`：

- `ppi_graph.py`：加载数据、标签和 split 图；
- `sampler.py`：按目标对生成可变长度子图轨迹；
- `predictor.py`：GAT 多标签边预测器；
- `trainer.py`：采样器与预测器的批量交替训练；
- `train_shs27k.py`：SHS27k/bfs 实验入口。

## 当前数据与接口约束

- 支持 `SHS27k`、`SHS148k`、`STRING`，split 按数据集定义校验。
- `PPIGraph.build_graph(split_name, remap_nodes=False, undirected=True)` 是训练器推荐输入，保持全局蛋白索引与特征行一致。
- 图拓扑只使用当前 split 的边。采样每个目标对前，必须移除目标边的两个方向，避免标签泄露。
- 采样器只使用节点嵌入和拓扑，不使用边标签；虚拟代理只存在于采样轨迹，两个目标节点可共享代理，预测基线图只有目标节点且无边。
- 虚拟代理加入初始子图后，会补齐已选节点之间的真实安全边；目标边始终不会加入子图。
- 采样决策因轨迹长度可变而按目标对逐个生成；`AlternatingTrainer` 对外只保留 `sampler_batch_step`、`predictor_batch_step` 和 `alternating_batch_step`，预测器计算与损失聚合采用 batch。
- `PPIPredictor.forward()` 返回 logits，训练使用 `BCEWithLogitsLoss`；`predict_proba()` 或评估逻辑返回 sigmoid 概率。
- 目标对表示使用 `u + v` 与 `abs(u - v)`，交换端点不改变预测结果。

## 已验证结果

- 标签的 7 种 mode 与 `PPIGraph.LABELS` 一致；actions 数据按无向蛋白对 OR 聚合。
- split 图默认补全反向边；采样器会移除目标边的正、反向形式。
- SHS27k/bfs 全量训练 10 epoch 已完成（CUDA、batch=32、hidden=256、GAT 2 层/4 heads、最大采样步数 3）。训练集 4,562 对，测试集 1,538 对。
- 最佳测试指标：Macro ROC-AUC 0.756130（epoch 7）、Micro ROC-AUC 0.787892（epoch 9）、Macro-F1 0.543118（epoch 8）、Micro-F1 0.593515（epoch 7）。逐 epoch 记录保存在 `shs27k_bfs_10epoch.json`。

追加实验：将 `max_steps` 从 3 提高到 10，在其余配置不变时完成 10 epoch，记录保存在 `shs27k_bfs_10epoch_maxsteps10.json`。`max_steps=10` 的最终训练 predictor loss / mean reward 为 0.122694 / 0.118102，较 `max_steps=3` 的 0.140889 / 0.069434 更好；但测试最佳 Macro/Micro ROC-AUC 为 0.743840/0.768853，最佳 Macro/Micro-F1 为 0.515162/0.577582，均低于 `max_steps=3` 的 0.756130/0.787892 和 0.543118/0.593515。当前证据表明更长轨迹降低训练损失并提高 reward，却未改善测试泛化，且训练耗时约从 530 秒增加到 1,181 秒。

修正虚拟代理初始化后，使用同一配置重新运行 `max_steps=10` 训练，结果保存在 `shs27k_bfs_10epoch_proxyfix.json`。修正后最终 predictor loss / mean reward 为 0.120313 / 0.113685（修正前为 0.122694 / 0.118102）；最佳 Macro/Micro ROC-AUC 为 0.725183/0.749146，最佳 Macro/Micro-F1 为 0.490072/0.529231，均低于修正前的 0.743840/0.768853 和 0.515162/0.577582。该单次实验说明修正改变了采样子图和训练轨迹，但未显示测试性能提升；需要多随机种子实验才能判断真实泛化影响。

## 维护说明

修改采样拓扑、预测输出或训练接口时，同时更新本文件与 `src/CLAUDE.md`，并运行最小 smoke test。不要依据错误的 Git 历史推断当前设计；以现有代码、数据集说明和实际验证为准。
