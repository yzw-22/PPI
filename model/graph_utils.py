"""
PPI 图构建与查询工具。

提供 ``PPIGraph`` 类用于高效存储和查询蛋白质相互作用网络，
以及 ``build_graph`` 函数从原始数据文件构建图结构。
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch


class PPIGraph:
    """蛋白质相互作用网络图。

    以邻接表形式存储无向 PPI 图，支持高效的前沿节点查找和
    边特征聚合（用于 Sampler 的注意力计算）。

    Attributes:
        num_nodes: 蛋白质节点总数。
        adj_list: 邻接表 ``adj_list[i] = [j, k, ...]``。
        edge_feat: 边特征字典，键为 ``(min(a,b), max(a,b))``，
                   值为 7 维 multi-hot ``Tensor``。
    """

    def __init__(
        self,
        num_nodes: int,
        adj_list: List[List[int]],
        edge_feat: Dict[Tuple[int, int], torch.Tensor],
    ):
        self.num_nodes = num_nodes
        self.adj_list = adj_list
        self.edge_feat = edge_feat


    def get_edge_feat(self, a: int, b: int) -> Optional[torch.Tensor]:
        """获取边 (a, b) 的 7 维 multi-hot 特征。

        Returns:
            7 维 Tensor，若边不存在则返回 ``None``。
        """
        key = (min(a, b), max(a, b))
        return self.edge_feat.get(key, None)

    def get_frontier(self, subgraph: Set[int]) -> List[int]:
        """获取子图的前沿节点集合。

        前沿 = 所有与子图中至少一个节点相邻、但自身不在子图中的节点。

        Args:
            subgraph: 当前子图节点集合。

        Returns:
            去重后的前沿节点列表。
        """
        frontier = set()
        for node in subgraph:
            for neighbor in self.adj_list[node]:
                if neighbor not in subgraph:
                    frontier.add(neighbor)
        return list(frontier)


# ---------------------------------------------------------------------------
# 标签名称 → 编号 映射
# ---------------------------------------------------------------------------

LABEL_TO_IDX = {
    "reaction": 0,
    "binding": 1,
    "ptmod": 2,
    "activation": 3,
    "inhibition": 4,
    "catalysis": 5,
    "expression": 6,
}

# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def build_graph(
    dataset_name: str,
    dataset_dir: str = "dataset",
    verbose: bool = True,
) -> PPIGraph:
    """从原始数据文件构建 ``PPIGraph``。

    处理流程:
        1. 读取 ``protein.{name}.sequences.dictionary.tsv`` → id ↔ idx 映射
        2. 读取 ``protein.actions.{name}.txt`` → (idx_a, idx_b) → multi-hot
        3. 读取 ``{name}_ppi_list.json`` → 邻接表 + 附加边特征

    Args:
        dataset_name: 数据集名称，如 ``"SHS27k"``, ``"SHS148k"``, ``"STRING"``。
        dataset_dir: 数据集根目录。
        verbose: 是否输出进度信息。

    Returns:
        构建好的 ``PPIGraph`` 实例。
    """
    base = Path(dataset_dir)

    seq_path = base / f"protein.{dataset_name}.sequences.dictionary.tsv"
    actions_path = base / f"protein.actions.{dataset_name}.txt"
    ppi_path = base / f"{dataset_name}_ppi_list.json"

    for p in [seq_path, actions_path, ppi_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")

    # ---- 1. 读取序列字典 → id_to_idx ----
    if verbose:
        print(f"[build_graph] Reading sequences from {seq_path} ...")
    id_to_idx: Dict[str, int] = {}
    with open(seq_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            id_to_idx[row["id"]] = i
    num_proteins = len(id_to_idx)
    if verbose:
        print(f"  → {num_proteins} proteins")

    # ---- 2. 读取 actions → 按蛋白对聚合 multi-hot ----
    if verbose:
        print(f"[build_graph] Reading actions from {actions_path} ...")
    pair_labels: Dict[Tuple[int, int], Set[int]] = {}

    # 使用分块读取处理大文件
    chunk_size = 500_000
    for chunk in pd.read_csv(actions_path, sep="\t", chunksize=chunk_size):
        for _, row in chunk.iterrows():
            id_a, id_b, mode = row["item_id_a"], row["item_id_b"], row["mode"]
            idx_a = id_to_idx.get(id_a)
            idx_b = id_to_idx.get(id_b)
            if idx_a is None or idx_b is None:
                continue
            if mode not in LABEL_TO_IDX:
                continue

            key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if key not in pair_labels:
                pair_labels[key] = set()
            pair_labels[key].add(LABEL_TO_IDX[mode])

    if verbose:
        print(f"  → {len(pair_labels)} unique protein pairs with labels")

    # ---- 3. 读取 PPI 列表 → 构建邻接表与边特征 ----
    if verbose:
        print(f"[build_graph] Reading PPI list from {ppi_path} ...")
    ppi_list: List[List[int]] = json.loads(ppi_path.read_text())
    if verbose:
        print(f"  → {len(ppi_list)} PPI pairs")

    adj_list: List[List[int]] = [[] for _ in range(num_proteins)]
    edge_feat: Dict[Tuple[int, int], torch.Tensor] = {}

    for a, b in ppi_list:
        # 邻接表（无向）
        adj_list[a].append(b)
        adj_list[b].append(a)

        # 边特征
        key = (min(a, b), max(a, b))
        if key not in edge_feat:  # PPI 列表可能含重复
            label_ids = pair_labels.get(key, set())
            multi_hot = torch.zeros(7, dtype=torch.float32)
            for lid in label_ids:
                multi_hot[lid] = 1.0
            edge_feat[key] = multi_hot

    graph = PPIGraph(
        num_nodes=num_proteins,
        adj_list=adj_list,
        edge_feat=edge_feat,
    )

    if verbose:
        total_edges = sum(len(adj) for adj in adj_list) // 2
        edges_with_labels = sum(1 for v in edge_feat.values() if v.sum() > 0)
        print(f"[build_graph] Done: {num_proteins} nodes, {total_edges} edges, "
              f"{edges_with_labels} edges with labels")

    return graph
