from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "rolling_cov",
    "topo_seriation_from_graph",
    "hrp_weights_from_order",
]


def rolling_cov(returns: pd.DataFrame, window: int = 60) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Compute rolling covariance matrices over a fixed-size window."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    data = returns.sort_index()
    covariances: Dict[pd.Timestamp, pd.DataFrame] = OrderedDict()

    for end_idx in range(window, len(data) + 1):
        window_slice = data.iloc[end_idx - window : end_idx]
        valid_cols = window_slice.count()
        window_slice = window_slice.loc[:, valid_cols >= window]
        if window_slice.shape[1] < 2:
            continue
        window_slice = window_slice.ffill().bfill()
        cov = window_slice.cov()
        if cov.isnull().values.any():
            cov = cov.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if cov.shape[0] < 2:
            continue
        covariances[data.index[end_idx - 1]] = cov

    return covariances


def topo_seriation_from_graph(cov: pd.DataFrame, graph: nx.Graph) -> List[str]:
    """Derive an asset order from a graph structure."""

    if cov.empty:
        return []

    assets = list(cov.columns)
    if graph is None or graph.number_of_nodes() == 0:
        return assets

    nodes = [node for node in graph.nodes if node in assets]
    if not nodes:
        return assets

    subgraph = graph.subgraph(nodes).copy()
    order: List[str] = []

    def component_score(component_nodes: Sequence[str]) -> float:
        subcov = cov.loc[component_nodes, component_nodes]
        return float(subcov.values.mean())

    for component in sorted(nx.connected_components(subgraph), key=lambda comp: component_score(list(comp))):
        comp_nodes = list(component)
        comp_subgraph = subgraph.subgraph(comp_nodes)
        start_node = max(comp_nodes, key=lambda node: cov.loc[node, node])
        order.extend(list(nx.dfs_preorder_nodes(comp_subgraph, source=start_node)))

    seen = set()
    serialised = [node for node in order if not (node in seen or seen.add(node))]
    missing = [asset for asset in assets if asset not in seen]
    serialised.extend(missing)
    return serialised


def hrp_weights_from_order(cov: pd.DataFrame, order: Sequence[str]) -> pd.Series:
    """Compute HRP weights given an ordered list of assets."""

    if cov.empty:
        raise ValueError("covariance matrix is empty")
    if not order:
        raise ValueError("order must contain at least one asset")

    order = [asset for asset in order if asset in cov.columns]
    if not order:
        raise ValueError("order does not match covariance columns")

    cov = cov.loc[order, order]
    weights = pd.Series(1.0, index=order, dtype=float)

    def cluster_variance(items: Sequence[str]) -> float:
        subcov = cov.loc[items, items].values
        if subcov.size == 1:
            return float(subcov[0, 0])
        w = np.full(len(items), 1.0 / len(items))
        return float(w @ subcov @ w)

    def recursive_bisect(items: Sequence[str]) -> None:
        if len(items) <= 1:
            return
        split = len(items) // 2
        left = items[:split]
        right = items[split:]
        var_left = cluster_variance(left)
        var_right = cluster_variance(right)
        denom = var_left + var_right
        if denom == 0:
            alloc_left = alloc_right = 0.5
        else:
            alloc_left = 1.0 - var_left / denom
            alloc_right = 1.0 - alloc_left
        weights[left] *= alloc_left
        weights[right] *= alloc_right
        recursive_bisect(left)
        recursive_bisect(right)

    recursive_bisect(order)
    total = weights.sum()
    if total == 0:
        raise ValueError("Resulting weights sum to zero")
    return (weights / total).rename("hrp_weight")
