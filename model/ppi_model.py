"""
组合模型（PPIModel）。

协调 SubgraphSampler 与 PPIPredictor 的交替训练：
- Sampler 阶段：固定 Predictor，用 REINFORCE 训练采样策略
  奖励 = l_0 - l_t（子图信息带来的性能提升）
- Predictor 阶段：固定 Sampler（argmax），用 BCE 监督训练 GNN
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

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

        固定 Predictor。对每条 PPI，Sampler 每扩展一个节点（每个时间步 t）
        即将当前子图 G_t 输入 Predictor（冻结）计算损失 l_t，得到该步奖励
        r_t = l_0 - l_t（基线损失 - 当前子图损失，正值表示引入子图信息后的提升），
        并逐时间步累积 REINFORCE 策略梯度损失与价值回归损失。

        Args:
            ppi_indices: 本步训练的 PPI 索引列表。

        Returns:
            含各项损失和奖励的字典。
        """
        self.sampler.train()
        self.predictor.eval()

        # 冻结 Predictor
        for p in self.predictor.parameters():
            p.requires_grad = False

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward = 0.0
        total_steps = 0

        for ppi_idx in ppi_indices:
            u, v = self.ppi_list[ppi_idx]
            label = self._get_label(ppi_idx).to(self.config.device)

            # ---- 计算基线损失 l_0（仅 {u, v} 的预测损失） ----
            with torch.no_grad():
                pred_0 = self.predictor(
                    self._esm_tensor, self.graph, [u, v], (u, v)
                )
                l_0 = F.binary_cross_entropy(pred_0, label)

            # ---- Sampler 前向（采样模式）：返回逐时间步轨迹 ----
            trajectory = self.sampler(
                self._esm_tensor, self.graph, u, v, training=True
            )

            # ---- 逐时间步：当前子图 G_t 输入 Predictor 计算损失与奖励 ----
            for step in trajectory.steps:
                with torch.no_grad():
                    pred_t = self.predictor(
                        self._esm_tensor,
                        self.graph,
                        step.subgraph_nodes,
                        (u, v),
                        edges=step.subgraph_edges,
                    )
                    l_t = F.binary_cross_entropy(pred_t, label)

                # 本步奖励 = 基线损失 - 当前子图损失（正值表示性能提升）
                reward = l_0 - l_t

                # REINFORCE 损失（Baseline 价值网络做方差缩减）
                advantage = reward.detach() - step.value.detach()
                policy_loss = -step.log_prob * advantage
                value_loss = F.mse_loss(
                    step.value, reward.detach().expand_as(step.value)
                )

                # 累积
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_reward += reward.item()
                total_steps += 1

                # 反向传播
                loss = (
                    policy_loss
                    + self.config.reinforce_baseline_coef * value_loss
                )
                loss.backward()

        # 梯度更新
        if total_steps > 0:
            self.opt_sampler.step()
            self.opt_sampler.zero_grad()

        # 解冻 Predictor
        for p in self.predictor.parameters():
            p.requires_grad = True

        return {
            "policy_loss": total_policy_loss / max(total_steps, 1),
            "value_loss": total_value_loss / max(total_steps, 1),
            "reward": total_reward / max(total_steps, 1),
        }

    def train_predictor_step(
        self,
        ppi_indices: List[int],
    ) -> Dict[str, float]:
        """执行一个 Predictor 训练步（有监督 BCE）。

        固定 Sampler（argmax 模式），用贪心采样的子图训练 GNN。
        优化目标为扩展子图的 BCE 监督损失 l_t。

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
                trajectory = self.sampler(
                    self._esm_tensor, self.graph, u, v, training=False
                )
            subgraph_nodes = trajectory.final_subgraph

            # Predictor 前向（使用 Sampler 保存的诱导边，保证结构一致）
            pred = self.predictor(
                self._esm_tensor,
                self.graph,
                subgraph_nodes,
                (u, v),
                edges=trajectory.final_edges,
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
    # 评估
    # ==================================================================

    @torch.no_grad()
    def _predict_matrix(
        self,
        ppi_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对 PPI 索引集合批量推理，返回预测概率矩阵与标签矩阵。

        Args:
            ppi_indices: PPI 索引列表。

        Returns:
            ``(y_pred, y_true)``，各为 ``[N, 7]`` 的 numpy 数组。
        """
        self.sampler.eval()
        self.predictor.eval()

        all_preds = []
        all_labels = []

        for ppi_idx in ppi_indices:
            u, v = self.ppi_list[ppi_idx]

            # 采样子图
            trajectory = self.sampler(
                self._esm_tensor, self.graph, u, v, training=False
            )
            subgraph_nodes = trajectory.final_subgraph

            # 预测（使用 Sampler 保存的诱导边）
            pred = self.predictor(
                self._esm_tensor,
                self.graph,
                subgraph_nodes,
                (u, v),
                edges=trajectory.final_edges,
            )  # [7]

            label = self._get_label(ppi_idx)  # [7]

            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())

        return np.stack(all_preds), np.stack(all_labels)

    @torch.no_grad()
    def tune_threshold(
        self,
        tune_indices: List[int],
        average: str = "macro",
        num_candidates: int = 100,
    ) -> float:
        """在给定集合上搜索使 F1 最大的全局决策阈值。

        F1 公式只定义在硬预测上，阈值属于决策规则的一部分而非公式本身。
        固定 0.5 仅在类别均衡、模型校准良好时才对应理论最优工作点；
        对类别不平衡的多标签问题，标准做法是在验证集上选择使 F1
        最大化的阈值，再将其应用于测试集评估。

        Args:
            tune_indices: 用于调阈值的样本（通常是验证集）。
            average: F1 的聚合方式（``'micro'`` 或 ``'macro'``）。
            num_candidates: 在 [0, 1] 上均匀搜索的候选阈值个数。

        Returns:
            使 F1 最大的全局阈值。
        """
        y_pred, y_true = self._predict_matrix(tune_indices)

        thresholds = np.linspace(0.01, 0.99, num_candidates)
        best_th, best_f1 = 0.5, -1.0
        for th in thresholds:
            y_pred_bin = (y_pred >= th).astype(np.int32)
            f1 = f1_score(y_true, y_pred_bin, average=average, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        return float(best_th)

    @torch.no_grad()
    def evaluate(
        self,
        ppi_indices: List[int],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """评估模型在给定 PPI 索引集合上的 AUC 和 F1-score。

        对每条 PPI 先采样扩展子图，再通过 Predictor 预测，
        汇总后计算多标签分类的 micro/macro AUC 和 F1。

        Args:
            ppi_indices: 待评估的 PPI 索引列表。
            threshold: F1 计算时的决策阈值（二值化 sigmoid 输出）。
                默认 0.5 只在类别均衡时理论最优；对类别不平衡的数据，
                应先用 ``tune_threshold`` 在验证集上选出最优阈值再评估。

        Returns:
            包含 ``auc_micro``, ``auc_macro``, ``f1_micro``, ``f1_macro`` 的字典。
        """
        y_pred, y_true = self._predict_matrix(ppi_indices)  # [N, 7]

        # AUC: 对每个标签独立计算，取 micro/macro 平均
        # 处理全 0 或全 1 标签（AUC 未定义的情况）
        auc_per_label = []
        for i in range(self.config.num_labels):
            if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
                auc_per_label.append(float('nan'))
            else:
                auc_per_label.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

        # micro AUC: 将所有标签展平后计算
        try:
            auc_micro = roc_auc_score(y_true.ravel(), y_pred.ravel())
        except ValueError:
            auc_micro = float('nan')

        # macro AUC: 各标签 AUC 的均值（忽略 NaN）
        valid_auc = [a for a in auc_per_label if not np.isnan(a)]
        auc_macro = float(np.mean(valid_auc)) if valid_auc else float('nan')

        # F1-score: 按理论公式 2·P·R/(P+R) 计算
        # 先在给定阈值下二值化 sigmoid 输出，再交给 sklearn 计算混淆矩阵指标
        y_pred_bin = (y_pred >= threshold).astype(np.int32)

        f1_micro = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)

        return {
            "auc_micro": auc_micro,
            "auc_macro": auc_macro,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        }

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

        trajectory = self.sampler(
            self._esm_tensor, self.graph, u, v, training=False
        )
        subgraph_nodes = trajectory.final_subgraph
        return self.predictor(
            self._esm_tensor,
            self.graph,
            subgraph_nodes,
            (u, v),
            edges=trajectory.final_edges,
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
