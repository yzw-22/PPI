"""
组合模型（PPIModel）。

协调 SubgraphSampler 与 PPIPredictor 的交替训练：
- Sampler 阶段：固定 Predictor，用 REINFORCE 训练采样策略
- Predictor 阶段：固定 Sampler（argmax），用 BCE 监督训练 GNN
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PPIConfig
from model.graph_utils import PPIGraph, build_graph
from model.predictor import PPIPredictor
from model.sampler import SubgraphSampler


class PPIModel:
    """PPI 预测组合模型。

    包含一个子图采样器和一个 GNN 预测器，支持交替训练和端到端推理。

    Parameters:
        config: 全局配置。
        graph: PPI 图（可选，若为 None 需调用 ``load_data`` 加载）。
        ppi_list: PPI 索引映射 ``[[a,b], ...]``（可选）。
    """

    def __init__(
        self,
        config: PPIConfig,
        graph: Optional[PPIGraph] = None,
        ppi_list: Optional[List[List[int]]] = None,
    ):
        self.config = config

        # 数据
        self.graph = graph
        self.ppi_list = ppi_list  # [[protein_a, protein_b], ...]

        # 模型
        self.sampler = SubgraphSampler(config).to(config.device)
        self.predictor = PPIPredictor(config).to(config.device)

        # 优化器
        self.opt_sampler = torch.optim.Adam(
            self.sampler.parameters(), lr=config.lr_sampler
        )
        self.opt_predictor = torch.optim.Adam(
            self.predictor.parameters(), lr=config.lr_predictor
        )

        # 状态
        self._esm_tensor: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None  # [num_ppi, 7]
        self._step_counter = 0

    # ==================================================================
    # 数据加载
    # ==================================================================

    @classmethod
    def from_dataset(
        cls,
        config: PPIConfig,
        dataset_name: str = "SHS27k",
        dataset_dir: str = "dataset",
        verbose: bool = True,
    ) -> "PPIModel":
        """从数据集文件构建模型（含图、PPI 列表和预计算标签）。

        Args:
            config: 模型配置。
            dataset_name: 数据集名称。
            dataset_dir: 数据集根目录。
            verbose: 是否输出进度。

        Returns:
            已加载数据的 ``PPIModel`` 实例。
        """
        # 构建图
        graph = build_graph(dataset_name, dataset_dir, verbose=verbose)

        # 加载 PPI 列表
        ppi_path = Path(dataset_dir) / f"{dataset_name}_ppi_list.json"
        ppi_list = json.loads(ppi_path.read_text())

        # 构建标签矩阵 [num_ppi, 7]
        if verbose:
            print(f"[PPIModel] Building label matrix for {len(ppi_list)} PPIs ...")
        labels = torch.zeros(len(ppi_list), config.num_labels)
        for ppi_idx, (a, b) in enumerate(ppi_list):
            feat = graph.get_edge_feat(a, b)
            if feat is not None:
                labels[ppi_idx] = feat

        model = cls(config, graph=graph, ppi_list=ppi_list)
        model._labels = labels

        if verbose:
            num_labeled = (labels.sum(dim=1) > 0).sum().item()
            print(f"[PPIModel] Loaded {dataset_name}: "
                  f"{len(ppi_list)} PPIs, {num_labeled} with labels")

        return model

    def set_esm_tensor(self, tensor_path: str):
        """加载预计算的 ESM 嵌入张量并移至设备。

        Args:
            tensor_path: ``*_tensor.pt`` 文件路径。
        """
        tensor = torch.load(tensor_path, map_location="cpu")
        self._esm_tensor = tensor.to(self.config.device)
        return self

    # ==================================================================
    # 训练
    # ==================================================================

    def train_sampler_step(
        self,
        ppi_indices: List[int],
    ) -> Dict[str, float]:
        """执行一个 Sampler 训练步（REINFORCE）。

        固定 Predictor，用随机策略采样子图，根据 Predictor 的 BCE 表现
        计算奖励并更新策略。

        Args:
            ppi_indices: 本步训练的 PPI 索引列表。

        Returns:
            含各项损失的字典。
        """
        self.sampler.train()
        self.predictor.eval()

        # 冻结 Predictor
        for p in self.predictor.parameters():
            p.requires_grad = False

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward = 0.0
        n = 0

        for ppi_idx in ppi_indices:
            u, v = self.ppi_list[ppi_idx]

            # Sampler 前向（采样模式）
            subgraph_nodes, log_prob, value = self.sampler(
                self._esm_tensor, self.graph, u, v, training=True
            )

            # Predictor 前向（无梯度）
            with torch.no_grad():
                pred = self.predictor(
                    self._esm_tensor, self.graph, subgraph_nodes, (u, v)
                )
                label = self._get_label(ppi_idx).to(pred.device)
                bce = F.binary_cross_entropy(pred, label)
                reward = -bce  # 奖励 = 负 BCE

            # REINFORCE 损失
            advantage = reward.detach() - value.detach()
            policy_loss = -log_prob * advantage
            value_loss = F.mse_loss(value, reward.detach().expand_as(value))

            # 累积
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_reward += reward.item()
            n += 1

            # 反向传播
            loss = policy_loss + self.config.reinforce_baseline_coef * value_loss
            loss.backward()

        # 梯度更新
        if n > 0:
            self.opt_sampler.step()
            self.opt_sampler.zero_grad()

        # 解冻 Predictor
        for p in self.predictor.parameters():
            p.requires_grad = True

        return {
            "policy_loss": total_policy_loss / max(n, 1),
            "value_loss": total_value_loss / max(n, 1),
            "reward": total_reward / max(n, 1),
        }

    def train_predictor_step(
        self,
        ppi_indices: List[int],
    ) -> Dict[str, float]:
        """执行一个 Predictor 训练步（有监督 BCE）。

        固定 Sampler（argmax 模式），用贪心采样的子图训练 GNN。

        Args:
            ppi_indices: 本步训练的 PPI 索引列表。

        Returns:
            含 BCE 损失的字典。
        """
        self.sampler.eval()
        self.predictor.train()

        total_loss = 0.0
        n = 0

        for ppi_idx in ppi_indices:
            u, v = self.ppi_list[ppi_idx]

            # Sampler 前向（argmax 模式，无梯度）
            with torch.no_grad():
                subgraph_nodes, _, _ = self.sampler(
                    self._esm_tensor, self.graph, u, v, training=False
                )

            # Predictor 前向
            pred = self.predictor(
                self._esm_tensor, self.graph, subgraph_nodes, (u, v)
            )
            label = self._get_label(ppi_idx).to(pred.device)
            loss = F.binary_cross_entropy(pred, label)

            total_loss += loss.item()
            n += 1

            loss.backward()

        # 梯度更新
        if n > 0:
            self.opt_predictor.step()
            self.opt_predictor.zero_grad()

        return {"bce_loss": total_loss / max(n, 1)}

    # ==================================================================
    # 推理
    # ==================================================================

    @torch.no_grad()
    def predict(self, u: int, v: int) -> torch.Tensor:
        """对单个 PPI 对进行推理。

        Args:
            u, v: 两个蛋白质的索引。

        Returns:
            7 维 Sigmoid 概率 ``[7]``。
        """
        self.sampler.eval()
        self.predictor.eval()

        subgraph_nodes, _, _ = self.sampler(
            self._esm_tensor, self.graph, u, v, training=False
        )
        return self.predictor(
            self._esm_tensor, self.graph, subgraph_nodes, (u, v)
        )

    @torch.no_grad()
    def predict_batch(
        self,
        pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """对一批 PPI 对进行推理。

        Args:
            pairs: ``[(u1, v1), (u2, v2), ...]``。

        Returns:
            预测概率矩阵 ``[B, 7]``。
        """
        results = []
        for u, v in pairs:
            results.append(self.predict(u, v))
        return torch.stack(results)

    # ==================================================================
    # Split 工具
    # ==================================================================

    def load_split(self, split_path: str) -> Dict[str, List[int]]:
        """加载训练/验证/测试划分。

        Args:
            split_path: ``*_bfs.json`` 等划分文件路径。

        Returns:
            ``{"train_index": [...], "val_index": [...], "test_index": [...]}``。
        """
        return json.loads(Path(split_path).read_text())

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _get_label(self, ppi_idx: int) -> torch.Tensor:
        """获取 PPI 索引对应的 7 维 multi-hot 标签。"""
        if self._labels is not None:
            return self._labels[ppi_idx].to(self.config.device)
        # fallback: 从图中实时查询
        a, b = self.ppi_list[ppi_idx]
        feat = self.graph.get_edge_feat(a, b)
        if feat is not None:
            return feat.to(self.config.device)
        return torch.zeros(self.config.num_labels, device=self.config.device)

    @property
    def esm_tensor(self) -> Optional[torch.Tensor]:
        return self._esm_tensor

    @property
    def labels(self) -> Optional[torch.Tensor]:
        return self._labels
