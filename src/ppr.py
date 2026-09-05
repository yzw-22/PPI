"""Sparse Personalized PageRank over the label-free knowledge graph.

The attention readout (``PPIPredictor(readout="attention")``) consumes
target-anchored positional encodings: for a sampled node ``n`` it uses
``[PPR_u(n), PPR_v(n)]`` — the importance of ``n`` under random walks with
teleport at each of the two target proteins.  Rows depend only on graph
topology (never on labels) and are computed lazily per target with the
standard forward-push algorithm, then cached for the lifetime of the lookup.
"""

from collections import deque

import torch


class PPRLookup:
    """Lazy, cached sparse PPR rows for every node of one undirected graph.

    Args:
        edge_index: ``[2, E]`` LongTensor of directed edge endpoints (both
            directions of an undirected graph).  Self-loops and parallel
            edges are collapsed.
        num_nodes: number of nodes; valid node ids are ``0..num_nodes-1``.
        alpha: teleport (restart) probability of the random walk.
        eps: forward-push accuracy threshold; a node is pushed when its
            residual reaches ``eps * degree``.  Smaller values give more
            accurate (denser) rows.
    """

    def __init__(self, edge_index, num_nodes, alpha=0.15, eps=5e-6):
        edge_index = torch.as_tensor(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        neighbors = [set() for _ in range(num_nodes)]
        for source, target in edge_index.t().tolist():
            if source != target:
                neighbors[source].add(target)
                neighbors[target].add(source)
        self.num_nodes = int(num_nodes)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._neighbors = tuple(tuple(sorted(n)) for n in neighbors)
        self._degree = tuple(len(n) for n in self._neighbors)
        self._cache = {}

    def get(self, target):
        """Return the sparse PPR row of ``target`` as ``{node: value}``.

        The dict is cached: repeated calls return the same object without
        recomputation.  Callers must not mutate it.
        """
        target = int(target)
        if not 0 <= target < self.num_nodes:
            raise ValueError(f"target {target} out of range")
        row = self._cache.get(target)
        if row is None:
            row = self._push(target)
            self._cache[target] = row
        return row

    def _push(self, seed):
        """Forward-push approximation of the PPR row of ``seed``.

        Standard local push: while a node's residual reaches
        ``eps * degree(node)``, move ``alpha`` of the residual into the
        estimate and distribute the rest uniformly over its neighbors.
        Deterministic; degree-0 nodes dump their residual into the estimate.
        """
        alpha, eps = self.alpha, self.eps
        neighbors, degree = self._neighbors, self._degree
        residual = {seed: 1.0}
        estimate = {}
        queue = deque([seed])
        queued = {seed}
        while queue:
            node = queue.popleft()
            queued.discard(node)
            value = residual.get(node, 0.0)
            deg = degree[node]
            if deg and value < eps * deg:
                continue
            estimate[node] = estimate.get(node, 0.0) + alpha * value
            if deg:
                share = (1.0 - alpha) * value / deg
                for neighbor in neighbors[node]:
                    pushed = residual.get(neighbor, 0.0) + share
                    residual[neighbor] = pushed
                    if (
                        pushed >= eps * degree[neighbor]
                        and neighbor not in queued
                    ):
                        queue.append(neighbor)
                        queued.add(neighbor)
            residual[node] = 0.0
        return estimate
