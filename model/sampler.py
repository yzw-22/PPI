"""
子图采样器（SubgraphSampler）。

从初始 PPI 对 (u, v) 出发，使用注意力机制逐步选择邻居节点扩展子图。
训练时通过 REINFORCE 策略梯度算法学习采样策略，推理时贪心选择。

邻居表征仅使用 ESM 蛋白质嵌入（不含边标签特征，避免数据泄露）。
对孤立/松散孤立 PPI，通过余弦相似度注入虚拟代理节点。
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PPIConfig
from model.graph_utils import PPIGraph


@dataclass
class SamplerStep:
    """一次扩展（一个时间步）的记录。

    Attributes:
        subgraph_nodes: 扩展后的子图 ``G_t`` 节点列表（含 ``u, v``、
            虚拟代理及本时间步新加入的节点）。
        subgraph_edges: ``G_t`` 的无向边列表（全局节点索引对，每对以
            ``(min, max)`` 规范化去重）。含虚拟代理时亦不含核心对
            ``(u, v)`` 的直连边（数据泄露防护）。
        log_prob: 本时间步动作的 log 概率（训练模式标量）。
        value: 本时间步动作前、由基线网络估计的状态价值 ``[1]``。
    """

    subgraph_nodes: List[int]
    subgraph_edges: List[Tuple[int, int]]
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass
class SamplerTrajectory:
    """一次完整的子图扩展轨迹。

    Attributes:
        initial_nodes: 初始子图 ``[u, v]``（含虚拟代理）。
        steps: 每个时间步的记录（仅训练模式填充）。
        final_subgraph: 实际扩展结束后的最终子图节点列表
            （推理模式 ``steps`` 为空时也能反映真实扩展结果）。
        final_edges: 最终子图的无向边列表（全局节点索引对，不含核心对
            ``(u, v)`` 的直连边）。
    """

    initial_nodes: List[int]
    steps: List[SamplerStep]
    final_subgraph: List[int]
    final_edges: List[Tuple[int, int]]


class SubgraphSampler(nn.Module):
    """可学习的子图采样器。

    每步扩展:
        1. 对孤立 PPI 对注入虚拟代理节点（可选）
        2. 计算当前子图状态 s_t（ESM 嵌入的 mean 池化）
        3. 对每个前沿邻居，使用其 ESM 嵌入作为邻居表征
        4. 通过缩放点积注意力计算选择概率分布
        5. 训练时依概率采样（REINFORCE），推理时选最大概率节点
        6. Baseline 网络估计状态价值，用于方差缩减

    Parameters:
        config: 全局配置对象。
    """

    def __init__(self, config: PPIConfig):
        super().__init__()
        self.config = config

        esm_dim = config.esm_dim
        hidden = config.hidden_dim
        attn_dim = config.attention_dim

        # 状态投影: esm_dim → hidden_dim
        self.state_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.ReLU(),
        )

        # 邻居表征 MLP: esm_dim → hidden_dim
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(esm_dim, hidden),
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
    ) -> SamplerTrajectory:
        """执行子图扩展，返回逐时间步的采样轨迹。

        每扩展一个节点（一个时间步）即记录该步的扩展后子图 ``G_t``、
        动作 log 概率与基线价值估计，供外部在训练时逐时间步调用
        Predictor 计算损失 ``l_t`` 与奖励 ``r_t = l_0 - l_t``。

        Args:
            esm_tensor: 蛋白质嵌入张量 ``[N, 2560]``（bfloat16 → 内部转为 float32）。
            graph: PPI 图。
            u, v: 初始 PPI 对的两个蛋白质索引。
            training: 是否训练模式（采样 vs argmax）。

        Returns:
            ``SamplerTrajectory``:
            - ``initial_nodes``: 初始子图 ``[u, v]``（含虚拟代理）。
            - ``steps``: 每个时间步的记录（仅训练模式填充，含扩展后子图、
              动作 log 概率与基线价值）。
            - ``final_subgraph``: 最终子图节点索引列表（无扩展时为初始子图）。
        """
        device = next(self.parameters()).device
        esm = esm_tensor.float().to(device)  # bfloat16 → float32, ensure device

        # ---- 初始子图 G_0 = {u, v} ----
        initial_nodes: List[int] = [u, v]

        # ---- 孤立 PPI 处理：注入虚拟代理 ----
        if self.config.isolated_proxy:
            proxy = self._find_virtual_proxy(esm, graph, u, v)
            if proxy is not None:
                initial_nodes.append(proxy)

        subgraph_nodes: List[int] = list(initial_nodes)
        subgraph_set: Set[int] = set(subgraph_nodes)

        # 子图边（无向，全局节点索引对，规范化为 (min, max)）。
        # 数据泄露防护：排除核心 PPI 对 (u, v) 的直连边——该边正是待预测的标签，
        # 若保留在子图拓扑中，模型可直接从结构读出答案。对训练集与测试集一致成立。
        # 初始：{u, v}（及虚拟代理）在原始图中实际存在的边（不含 (u, v) 本身）。
        core_edge = (min(u, v), max(u, v))
        subgraph_edges: List[Tuple[int, int]] = []
        for a in initial_nodes:
            for b in graph.adj_list[a]:
                if b in subgraph_set and a < b:
                    if (a, b) == core_edge:
                        continue
                    subgraph_edges.append((a, b))

        steps: List[SamplerStep] = []

        for _ in range(self.config.T_max):
            # 寻找前沿
            frontier = graph.get_frontier(subgraph_set)
            if not frontier:
                break

            # 计算状态
            state = self._compute_state(esm, subgraph_nodes)  # [esm_dim]

            # 计算邻居表征（仅使用 ESM 嵌入）
            neighbor_reprs = self._compute_neighbor_reprs(
                esm, frontier
            )  # [F, hidden_dim]

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
            else:
                selected_idx = probs.argmax(dim=-1)
                log_prob = None

            new_node = frontier[selected_idx.item()]
            subgraph_nodes.append(new_node)
            subgraph_set.add(new_node)

            # 新增边：新节点与已有子图内邻居之间的图边
            # （每个节点只进入子图一次，故每条边恰好由较晚进入的一端触发一次）。
            # 新节点来自前沿（不含 u, v），故此处不可能产生核心对 (u, v) 直连边；
            # 仍加防御性跳过以保证不变量在任何路径下成立。
            for nbr in graph.adj_list[new_node]:
                if nbr in subgraph_set:
                    a, b = (min(new_node, nbr), max(new_node, nbr))
                    if (a, b) == core_edge:
                        continue
                    subgraph_edges.append((a, b))

            # 记录本时间步（扩展后子图 G_t 的节点+边，及该步动作信息）
            if training:
                steps.append(
                    SamplerStep(
                        subgraph_nodes=list(subgraph_nodes),
                        subgraph_edges=list(subgraph_edges),
                        log_prob=log_prob,
                        value=value,
                    )
                )

        return SamplerTrajectory(
            initial_nodes=initial_nodes,
            steps=steps,
            final_subgraph=list(subgraph_nodes),
            final_edges=list(subgraph_edges),
        )

    # ------------------------------------------------------------------
    # 虚拟代理注入（孤立 PPI 处理）
    # ------------------------------------------------------------------

    def _find_virtual_proxy(
        self,
        esm: torch.Tensor,
        graph: PPIGraph,
        u: int,
        v: int,
    ) -> Optional[int]:
        """为孤立/松散孤立 PPI 对寻找虚拟代理节点。

        松散孤立定义：至少一端 degree = 1。
        虚拟代理 = 与 u 或 v 余弦相似度最大的节点（排除 u, v 自身及已连接邻居）。

        Args:
            esm: 蛋白质嵌入 ``[N, 2560]``（float32）。
            graph: PPI 图。
            u, v: 核心 PPI 对。

        Returns:
            虚拟代理节点索引，若非孤立 PPI 则返回 ``None``。
        """
        deg_u = len(graph.adj_list[u])
        deg_v = len(graph.adj_list[v])

        # 非松散孤立：两端度均 > 1
        if deg_u > 1 and deg_v > 1:
            return None

        # 需排除的节点：u, v 自身, 以及它们的直接邻居
        exclude = {u, v}
        exclude.update(graph.adj_list[u])
        exclude.update(graph.adj_list[v])

        N = esm.shape[0]
        if len(exclude) >= N:
            return None  # 没有可选节点

        # 归一化嵌入
        esm_norm = F.normalize(esm, dim=-1)  # [N, 2560]

        u_emb = esm_norm[u]  # [2560]
        v_emb = esm_norm[v]  # [2560]

        # 与所有节点的余弦相似度
        sim_u = torch.matmul(esm_norm, u_emb)  # [N]
        sim_v = torch.matmul(esm_norm, v_emb)  # [N]

        # 取 u 和 v 中相似度的最大值
        sim_max = torch.max(sim_u, sim_v)  # [N]

        # 排除已连接节点
        for idx in exclude:
            if 0 <= idx < N:
                sim_max[idx] = -float('inf')

        best = sim_max.argmax().item()
        if sim_max[best].item() <= -1e9:
            return None

        return best

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
        frontier: List[int],
    ) -> torch.Tensor:
        """计算所有前沿邻居的表征向量（仅使用 ESM 嵌入）。

        Args:
            esm: 蛋白质嵌入 ``[N, 2560]``（float32）。
            frontier: 前沿节点索引列表。

        Returns:
            邻居表征矩阵 ``[F, hidden_dim]``。
        """
        indices = torch.tensor(frontier, device=esm.device, dtype=torch.long)
        esm_feats = esm.index_select(0, indices)  # [F, esm_dim]
        return self.neighbor_mlp(esm_feats)  # [F, hidden_dim]

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
