"""
PPI 模型配置模块。

集中管理 SubgraphSampler、PPIPredictor 以及交替训练的所有超参数。
"""

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class PPIConfig:
    """PPI 模型全局配置。

    Attributes:
        esm_dim: ESM-2 3B 蛋白质嵌入维度。
        hidden_dim: 通用隐层维度（Sampler 和 Predictor 共享）。
        attention_dim: Sampler 中 Q/K 注意力投影维度。
        T_max: 子图最大扩展步数（最终子图最多 2 + T_max 个节点）。
        gnn_num_layers: Predictor GNN 层数。
        gnn_dropout: GNN 层 dropout 比率。
        gnn_heads: GAT 注意力头数。
        num_labels: PPI 关系类别数（7）。
        num_edge_types: 边特征维度（7 类 multi-hot）。
        lr_sampler: Sampler 学习率（REINFORCE）。
        lr_predictor: Predictor 学习率（BCE 监督）。
        sampler_steps: 每次交替中 Sampler 的训练步数。
        predictor_steps: 每次交替中 Predictor 的训练步数。
        reinforce_baseline_coef: REINFORCE baseline 损失的权重系数 β。
        use_edge_features_in_sampler: Sampler 是否使用邻居边的 multi-hot
            标签作为特征。设为 False 则仅使用二值连接指示器。
        device: 计算设备。
    """

    # --- 维度 ---
    esm_dim: int = 2560
    hidden_dim: int = 256
    attention_dim: int = 64

    # --- Sampler ---
    T_max: int = 10
    use_edge_features_in_sampler: bool = True

    # --- Predictor (GNN) ---
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.3
    gnn_heads: int = 4

    # --- 标签 ---
    num_labels: int = 7
    num_edge_types: int = 7

    # --- 训练 ---
    lr_sampler: float = 1e-4
    lr_predictor: float = 1e-3
    sampler_steps: int = 5
    predictor_steps: int = 5
    reinforce_baseline_coef: float = 0.5
    grad_accum_steps: int = 4  # 梯度累积步数（缓解逐样本无法批量化的问题）

    # --- 设备 ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- 标签名称 ---
    LABEL_NAMES: tuple = field(
        default=("reaction", "binding", "ptmod", "activation", "inhibition", "catalysis", "expression"),
        init=False,
        repr=False,
    )

    def __post_init__(self):
        if self.pooling_mode not in ("mean", "sum"):
            raise ValueError(f"pooling_mode must be 'mean' or 'sum', got '{self.pooling_mode}'")
