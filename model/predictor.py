"""
GNN 预测器（PPIPredictor）。

对 Sampler 扩展后的子图运行图神经网络，输出 7 维多标签概率。
使用 GAT（Graph Attention Network）作为消息传递主干。
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from model.config import PPIConfig
from model.graph_utils import PPIGraph


class PPIPredictor(nn.Module):
    """基于 GNN 的 PPI 关系预测器。

    对输入的扩展子图:
        1. 将 ESM 嵌入投影到隐层空间
        2. 通过多层 GAT 进行消息传递
        3. 提取核心对 (u, v) 的表示
        4. 通过 MLP 输出 7 维 Sigmoid 概率

    Parameters:
        config: 全局配置对象。
    """

    def __init__(self, config: PPIConfig):
        super().__init__()
        self.config = config

        hidden = config.hidden_dim
        heads = config.gnn_heads
        out_per_head = hidden // heads  # 每个头的输出维度
        dropout = config.gnn_dropout

        # 节点投影
        self.node_proj = nn.Sequential(
            nn.Linear(config.esm_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GAT 层 + LayerNorm + 残差
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.gnn_num_layers):
            self.convs.append(
                GATConv(
                    hidden, out_per_head,
                    heads=heads,
                    dropout=dropout,
                    concat=True,  # concat all heads → hidden dim
                )
            )
            self.norms.append(nn.LayerNorm(hidden))

        # 分类器：pairwise readout → MLP → 7
        # readout: [u; v; u⊙v; |u−v|] → 4 * hidden
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, config.num_labels),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        esm_tensor: torch.Tensor,
        graph: PPIGraph,
        subgraph_nodes: List[int],
        core_pair: Tuple[int, int],
    ) -> torch.Tensor:
        """预测 PPI 关系的 7 维概率。

        Args:
            esm_tensor: 蛋白质嵌入张量 ``[N, 2560]``。
            graph: PPI 图。
            subgraph_nodes: 子图中所有节点索引列表（含 u, v）。
            core_pair: 核心 PPI 对 ``(u, v)``。

        Returns:
            7 维 Sigmoid 概率 Tensor ``[7]``。
        """
        device = next(self.parameters()).device
        esm = esm_tensor.float().to(device)  # bfloat16 → float32, ensure device

        # 构建子图 PyG Data
        x, edge_index, u_local, v_local = self._build_subgraph(
            esm, graph, subgraph_nodes, core_pair
        )

        # 节点投影
        h = self.node_proj(x)  # [N_sub, hidden]

        # GNN 消息传递
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)  # [N_sub, hidden]
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.config.gnn_dropout, training=self.training)
            h = h + h_new  # 残差连接

        # Pairwise readout
        h_u = h[u_local]  # [hidden]
        h_v = h[v_local]  # [hidden]

        pairwise = torch.cat([
            h_u,
            h_v,
            h_u * h_v,
            torch.abs(h_u - h_v),
        ], dim=-1)  # [4 * hidden]

        logits = self.classifier(pairwise)  # [7]
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_subgraph(
        self,
        esm: torch.Tensor,
        graph: PPIGraph,
        subgraph_nodes: List[int],
        core_pair: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """从原始图中提取导出子图，构建 PyG 输入。

        Args:
            esm: 蛋白质嵌入 ``[N, 2560]``（float32）。
            graph: PPI 图。
            subgraph_nodes: 子图节点列表。
            core_pair: 核心对 (u, v)。

        Returns:
            ``(x, edge_index, u_local, v_local)``
            - x: 节点特征 ``[N_sub, esm_dim]``
            - edge_index: 边索引 ``[2, E]``（双向）
            - u_local: 核心节点 u 在子图中的局部索引
            - v_local: 核心节点 v 在子图中的局部索引
        """
        u, v = core_pair

        # 全局 → 局部索引映射
        node_to_local = {n: i for i, n in enumerate(subgraph_nodes)}
        u_local = node_to_local[u]
        v_local = node_to_local[v]

        # 节点特征
        indices = torch.tensor(subgraph_nodes, device=esm.device, dtype=torch.long)
        x = esm.index_select(0, indices)  # [N_sub, esm_dim]

        # 构建边（无向 → 双向）
        edge_pairs = []
        for node in subgraph_nodes:
            for neighbor in graph.adj_list[node]:
                if neighbor in node_to_local and node < neighbor:
                    li = node_to_local[node]
                    lj = node_to_local[neighbor]
                    edge_pairs.append([li, lj])

        if edge_pairs:
            ei = torch.tensor(edge_pairs, device=esm.device, dtype=torch.long).t()
            ei = torch.cat([ei, ei.flip(0)], dim=-1)  # 双向边
        else:
            ei = torch.zeros((2, 0), device=esm.device, dtype=torch.long)

        return x, ei, u_local, v_local
