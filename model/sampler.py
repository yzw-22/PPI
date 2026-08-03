"""
子图采样器（SubgraphSampler）。

从初始 PPI 对 (u, v) 出发，使用注意力机制逐步选择邻居节点扩展子图。
训练时通过 REINFORCE 策略梯度算法学习采样策略，推理时贪心选择。
"""

import math
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PPIConfig
from model.graph_utils import PPIGraph


class SubgraphSampler(nn.Module):
    """可学习的子图采样器。

    每步扩展:
        1. 计算当前子图状态 s_t（ESM 嵌入的 mean/sum 池化）
        2. 对每个前沿邻居，结合其 ESM 嵌入和到子图的边特征，形成邻居表征
        3. 通过缩放点积注意力计算选择概率分布
        4. 训练时依概率采样（REINFORCE），推理时选最大概率节点
        5. Baseline 网络估计状态价值，用于方差缩减

    Parameters:
        config: 全局配置对象。
    """

    def __init__(self, config: PPIConfig):
        super().__init__()
        self.config = config

        esm_dim = config.esm_dim
        hidden = config.hidden_dim
        attn_dim = config.attention_dim
        edge_dim = config.num_edge_types

        # 状态投影: esm_dim → hidden_dim
        self.state_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.ReLU(),
        )

        # 邻居表征 MLP: [esm_dim + edge_dim] → hidden_dim
        neighbor_in_dim = esm_dim + edge_dim
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(neighbor_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # 注意力投影: hidden_dim → attention_dim
        self.query_proj = nn.Linear(hidden, attn_dim)
        self.key_proj = nn.Linear(hidden, attn_dim)

        # Baseline 价值网络: hidden_dim → 1
        self.baseline = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        self._temperature = math.sqrt(attn_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        esm_tensor: torch.Tensor,
        graph: PPIGraph,
        u: int,
        v: int,
        training: bool = True,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """执行子图扩展。

        Args:
            esm_tensor: 蛋白质嵌入张量 ``[N, 2560]``（bfloat16 → 内部转为 float32）。
            graph: PPI 图。
            u, v: 初始 PPI 对的两个蛋白质索引。
            training: 是否训练模式（采样 vs argmax）。

        Returns:
            ``(subgraph_nodes, log_prob_mean, value_mean)``
            - subgraph_nodes: 最终子图节点索引列表。
            - log_prob_mean: 平均对数概率（仅训练模式有效，推理模式为 0）。
            - value_mean: 平均 baseline 估计值。
        """
        device = next(self.parameters()).device
        esm = esm_tensor.float().to(device)  # bfloat16 → float32, ensure device

        subgraph_nodes: List[int] = [u, v]
        subgraph_set: Set[int] = {u, v}

        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []

        for _ in range(self.config.T_max):
            # 寻找前沿
            frontier = graph.get_frontier(subgraph_set)
            if not frontier:
                break

            # 计算状态
            state = self._compute_state(esm, subgraph_nodes)  # [esm_dim]

            # 计算邻居表征
            neighbor_reprs = self._compute_neighbor_reprs(
                esm, graph, frontier, subgraph_set
            )  # [F, hidden]

            # 注意力得分
            scores = self._attention_scores(state, neighbor_reprs)  # [F]
            probs = F.softmax(scores, dim=-1)  # [F]

            # Baseline 价值估计
            state_hidden = self.state_proj(state.unsqueeze(0))  # [1, hidden]
            value = self.baseline(state_hidden).squeeze(-1)  # [1]

            # 选择节点
            if training:
                dist = torch.distributions.Categorical(probs)
                selected_idx = dist.sample()
                log_prob = dist.log_prob(selected_idx)
                log_probs.append(log_prob)
                values.append(value)
            else:
                selected_idx = probs.argmax(dim=-1)

            new_node = frontier[selected_idx.item()]
            subgraph_nodes.append(new_node)
            subgraph_set.add(new_node)

        # 汇总
        if log_probs:
            log_prob_mean = torch.stack(log_probs).mean()
            value_mean = torch.stack(values).mean()
        else:
            # 无前沿邻居时，返回与模型参数图连通的零张量（保证 requires_grad=True）
            p0 = next(iter(self.parameters()))
            dummy = p0.flatten()[0] * 0.0
            log_prob_mean = dummy
            value_mean = dummy

        return subgraph_nodes, log_prob_mean, value_mean

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_state(
        self,
        esm: torch.Tensor,
        subgraph_nodes: List[int],
    ) -> torch.Tensor:
        """计算当前子图状态向量。

        Args:
            esm: 蛋白质嵌入 ``[N, 2560]``（float32）。
            subgraph_nodes: 子图中节点索引列表。

        Returns:
            状态向量 ``[esm_dim]``。
        """
        indices = torch.tensor(subgraph_nodes, device=esm.device, dtype=torch.long)
        embeddings = esm.index_select(0, indices)  # [|G_t|, esm_dim]
        return embeddings.mean(dim=0)  # [esm_dim]


    def _compute_neighbor_reprs(
        self,
        esm: torch.Tensor,
        graph: PPIGraph,
        frontier: List[int],
        subgraph_set: Set[int],
    ) -> torch.Tensor:
        """计算所有前沿邻居的表征向量。

        Args:
            esm: 蛋白质嵌入 ``[N, 2560]``。
            graph: PPI 图。
            frontier: 前沿节点索引列表。
            subgraph_set: 当前子图节点集合。

        Returns:
            邻居表征矩阵 ``[F, hidden_dim]``。
        """
        # 收集 ESM 嵌入
        indices = torch.tensor(frontier, device=esm.device, dtype=torch.long)
        esm_feats = esm.index_select(0, indices)  # [F, esm_dim]

        # 收集聚合边特征
        edge_feats_list = []
        for node in frontier:
            agg = graph.aggregate_edge_features(
                node, subgraph_set,
                use_edge_features=self.config.use_edge_features_in_sampler,
            )
            edge_feats_list.append(agg)

        edge_feats = torch.stack(edge_feats_list).to(device=esm.device, dtype=esm.dtype)  # [F, 7]

        # 拼接并通过 MLP
        neighbor_input = torch.cat([esm_feats, edge_feats], dim=-1)  # [F, esm_dim+7]
        return self.neighbor_mlp(neighbor_input)  # [F, hidden_dim]

    def _attention_scores(
        self,
        state: torch.Tensor,
        neighbor_reprs: torch.Tensor,
    ) -> torch.Tensor:
        """计算状态与各邻居表征之间的注意力得分。

        Args:
            state: 状态向量 ``[esm_dim]``。
            neighbor_reprs: 邻居表征 ``[F, hidden_dim]``。

        Returns:
            注意力得分 ``[F]``（未归一化）。
        """
        state_hidden = self.state_proj(state.unsqueeze(0))  # [1, hidden]
        query = self.query_proj(state_hidden)  # [1, attention_dim]
        keys = self.key_proj(neighbor_reprs)  # [F, attention_dim]

        # 缩放点积
        scores = (query * keys).sum(dim=-1) / self._temperature  # [F]
        return scores
