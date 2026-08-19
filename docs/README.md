# docs 目录总结报告

> 生成于 readout/超参数调优实验收官后。本文件是 docs 目录的索引与精简说明。

## 目录结构与定位

| 文档 | 定位 | 精简后行数 |
|---|---|---|
| [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) | **当前实验总结**：readout 三种模式与超参数调优的全部结论、12 次运行总表、建议基线命令 | 76 |
| [TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md) | 训练性能瓶颈与优化方案（由仓库根目录移入并精简） | 73 |
| [SHS27k_BFS_KHOPS1_REINFORCE_GAMMA1_EPOCH10.md](SHS27k_BFS_KHOPS1_REINFORCE_GAMMA1_EPOCH10.md) | 历史单次实验记录（当前 k_hops 语义，gamma=1.0/10ep） | 45 |
| [SHS27k_BFS_REINFORCE_GAMMA1_EPOCH10.md](SHS27k_BFS_REINFORCE_GAMMA1_EPOCH10.md) | 历史单次实验记录（旧 G0/初始邻居机制，当前 CLI 不可复现） | 39 |

## 核心结论速览（详见 EXPERIMENT_SUMMARY.md）

- **当前最优配置**：`v1 + hidden 128 + gamma 0.9 + 20 epochs`，seed 42：
  test MacF1 0.5090 / MicF1 0.5399 / MacAUC 0.7275 / MicAUC 0.7315。
- **readout**：纯 ⊙（attn）明显差；补回 |u−v|、u+v（attn2）回升但均值仍略
  低于 v1（差距在方差内）→ v1 保持默认。
- **超参**：gamma 0.9 > 1.0（v1）；hidden 128 > 256（小数据域降过拟合，
  提速 ~35%）。
- **纪律**：指标方差大（MacF1 0.36~0.51），结论须多 seed 均值±std；
  推理特征禁止含任何标签信息。

## 精简说明

- `TRAINING_OPTIMIZATION.md` 由根目录移入 docs（`git mv`，根 `CLAUDE.md`
  链接已同步更新），119 → 73 行：压缩瓶颈/实施顺序的描述文字，保留
  优先级表、profiling 项与"必须保持的不变量"。
- 两份历史实验记录删除完整训练曲线表与冗余限制，95/108 → 45/39 行，
  标注历史性并链接新总结。
- 四份核心文档总量：原 203（两报告）+ 119（优化文档）= 322 行 → 现 233 行
  （−28%），信息密度提升，核心数据无丢失。

## 更新约定

- 新实验结论只写进 `EXPERIMENT_SUMMARY.md`（保持单一事实源）；
- 历史文档不再追加新内容，仅保留可复现配置与最佳结果；
- 训练稳定性/泄漏红线以 `TRAINING_OPTIMIZATION.md` 与 `src/CLAUDE.md` 为准。
