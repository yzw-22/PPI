"""
PPI 模型配置模块。

集中管理 SubgraphSampler、PPIPredictor 以及交替训练的所有超参数。
"""

from dataclasses import dataclass, field

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
        lr_sampler: Sampler 学习率（REINFORCE）。
        lr_predictor: Predictor 学习率（BCE 监督）。
        sampler_steps: 每次交替中 Sampler 的训练步数。
        predictor_steps: 每次交替中 Predictor 的训练步数。
        reinforce_baseline_coef: REINFORCE baseline 损失的权重系数 β。
        isolated_proxy: 是否为孤立/松散孤立 PPI 注入虚拟代理节点。
        device: 计算设备。
    """

    # --- 维度 ---
    esm_dim: int = 2560
    hidden_dim: int = 256
    attention_dim: int = 64

    # --- Sampler ---
    T_max: int = 10
    isolated_proxy: bool = True  # 是否为孤立/松散孤立 PPI 注入虚拟代理

    # --- Predictor (GNN) ---
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.3
    gnn_heads: int = 4
    # 残差连接：h = h + gnn_residual_scale * h_new（诊断"残差是否稀释消息传递"）
    gnn_residual: bool = True
    gnn_residual_scale: float = 1.0

    # --- 标签 ---
    num_labels: int = 7

    # --- 训练 ---
    lr_sampler: float = 1e-4
    lr_predictor: float = 1e-3
    sampler_steps: int = 5
    predictor_steps: int = 5
    reinforce_baseline_coef: float = 0.5
    
    # --- 设备 ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- 标签名称 ---
    LABEL_NAMES: tuple = field(
        default=("reaction", "binding", "ptmod", "activation", "inhibition", "catalysis", "expression"),
        init=False,
        repr=False,
    )
    