"""
PPI (Protein-Protein Interaction) 预测模型。

包含：
- SubgraphSampler:  可学习的子图采样器（注意力 + REINFORCE RL）
- PPIPredictor:     GNN 多标签预测器
- PPIModel:         组合模型（交替训练调度）
- PPIGraph:         PPI 图数据结构
- PPIConfig:        全局配置
"""

from model.config import PPIConfig
from model.graph_utils import PPIGraph, build_graph
from model.predictor import PPIPredictor
from model.sampler import SubgraphSampler
from model.ppi_model import PPIModel

__all__ = [
    "PPIConfig",
    "PPIGraph",
    "build_graph",
    "SubgraphSampler",
    "PPIPredictor",
    "PPIModel",
]
