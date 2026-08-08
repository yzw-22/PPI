# PPI 项目现状

本项目使用预计算的 ESM-2 3B 嵌入预测 7 类 PPI 多标签关系。数据格式见 [dataset/CLAUDE.md](dataset/CLAUDE.md)，模型细节见 [src/CLAUDE.md](src/CLAUDE.md)。

## 代码结构

- `ppi_graph.py`：加载 SHS27k、SHS148k、STRING，聚合标签并构建 split 图。
- `sampler.py`：为每个目标 PPI 生成 REINFORCE 子图轨迹。
- `predictor.py`：使用 GAT 对采样子图进行多标签预测。
- `trainer.py`：按 batch 交替更新 Sampler 和 Predictor。
- `train_shs27k.py`：SHS27k/bfs 训练与评估入口。

## 核心约束

- 每次训练和评估只使用当前 split 的边；目标边 `(u,v)` 和 `(v,u)` 在采样前移除。
- 训练器使用 `build_graph(split_name, remap_nodes=False)`，保持全局蛋白索引与特征行一致。
- 基线图仅含目标节点且无边。孤立目标可使用余弦相似度最高的虚拟代理；代理不能是目标节点，两个目标可共享代理。
- 代理加入后补齐已选节点之间的真实安全边。虚拟边和真实边都进入 Predictor，当前没有 edge type。
- Sampler 按目标逐条生成可变长度轨迹；Predictor 将多个子图拼接后批量计算。
- 每个 batch 先更新 Sampler，再更新 Predictor。训练 PPI 对在每个 epoch 通过 `torch.randperm` 重新打乱。
- Predictor 内部输出 logits，训练使用 BCE with logits；评估输出 sigmoid 概率。
- 目标对表示采用 `u+v` 与 `|u-v|`，Predictor readout 对端点交换保持不变。

## 当前训练语义

- Reward：`r_t = baseline_loss - step_loss`。
- Sampler loss：`policy_loss + β × value_loss`。
- Reward 和损失按所有 trajectory step 平均，不按子图大小或每条 trajectory 单独归一化。
- `max_steps` 限制新增节点的动作数，不限制图的 hop 深度；Sampler 类默认值为 10，训练脚本命令行默认值为 3。

## 当前实验结果

当前代码在 SHS27k/bfs 上完成了 `max_steps=10`、10 epoch 全量训练：CUDA、seed 42、batch size 32、hidden 256、2 层 GAT、4 heads。训练集 4,562 对，测试集 1,538 对。

- 最终 Sampler loss：0.005087
- 最终 Predictor loss：0.120313
- 最终 mean reward：0.113685
- 最佳 Macro ROC-AUC：0.725183（epoch 10）
- 最佳 Micro ROC-AUC：0.749146（epoch 8）
- 最佳 Macro-F1：0.490072（epoch 3）
- 最佳 Micro-F1：0.529231（epoch 3）

完整记录见 `shs27k_bfs_10epoch_proxyfix.json`。这是单个随机种子的结果。

## 验证

```bash
python -m unittest discover -s tests -v
python -m src.train_shs27k --max-steps 10
```
