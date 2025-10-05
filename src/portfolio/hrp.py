from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "rolling_cov",
    "topo_seriation_from_graph",
    "hrp_weights_from_order",
]


def rolling_cov(
    returns: pd.DataFrame,
    window: int = 60,
    method_cfg: Optional[Dict[str, Any]] = None,
    target_dates: Optional[Iterable[pd.Timestamp]] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Compute rolling covariance matrices with optional EWMA and shrinkage.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide returns matrix indexed by date.
    window : int, default 60
        Rolling lookback size in rows.
    method_cfg : dict, optional
        Optional configuration for EWMA/shrinkage.
    target_dates : iterable of Timestamp, optional
        Restrict computation to a subset of dates (e.g., rebalance days). When
        ``None`` (default) the covariance is computed for every available date.
    """

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    cfg = method_cfg or {}
    ewma_lambda = cfg.get("ewma_lambda")
    shrinkage = str(cfg.get("shrinkage", "")).strip().lower() or None
    if ewma_lambda is not None:
        ewma_lambda = float(ewma_lambda)
        if not (0.0 < ewma_lambda < 1.0):
            raise ValueError("ewma_lambda must lie in (0, 1)")
        if shrinkage == "ledoit_wolf":
            raise ValueError("Cannot combine ewma_lambda with shrinkage='ledoit_wolf'")

    data = returns.sort_index()
    if target_dates is not None:
        target_index = pd.Index(pd.to_datetime(list(target_dates))).unique().sort_values()
        target_set = set(target_index)
    else:
        target_set = None
    covariances: Dict[pd.Timestamp, pd.DataFrame] = OrderedDict()

    for end_idx in range(window, len(data) + 1):
        window_slice = data.iloc[end_idx - window : end_idx]
        current_date = data.index[end_idx - 1]
        if target_set is not None and current_date not in target_set:
            continue
        valid_cols = window_slice.count()
        window_slice = window_slice.loc[:, valid_cols >= window]
        if window_slice.shape[1] < 2:
            continue
        window_slice = window_slice.ffill().bfill()

        if ewma_lambda is not None:
            cov = _ewma_covariance(window_slice, ewma_lambda)
        elif shrinkage == "ledoit_wolf":
            cov = _ledoit_wolf_covariance(window_slice)
        else:
            cov = window_slice.cov()

        if cov.isnull().values.any():
            cov = cov.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if cov.shape[0] < 2:
            continue

        if shrinkage and shrinkage != "ledoit_wolf":
            cov = _apply_shrinkage(cov, shrinkage, cfg)

        covariances[data.index[end_idx - 1]] = cov

    return covariances


def _ewma_covariance(window_slice: pd.DataFrame, lam: float) -> pd.DataFrame:
    values = window_slice.to_numpy(dtype=float)
    n_obs = values.shape[0]
    weights = lam ** np.arange(n_obs)[::-1]
    weights = weights / weights.sum()
    mean = np.average(values, axis=0, weights=weights)
    demeaned = values - mean
    weighted = demeaned * weights[:, None]
    cov_matrix = weighted.T @ demeaned
    return pd.DataFrame(cov_matrix, index=window_slice.columns, columns=window_slice.columns)


def _ledoit_wolf_covariance(window_slice: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(window_slice.to_numpy(dtype=float))
    cov_matrix = lw.covariance_
    return pd.DataFrame(cov_matrix, index=window_slice.columns, columns=window_slice.columns)


def _apply_shrinkage(cov: pd.DataFrame, shrinkage: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    shrinkage = shrinkage.lower()
    strength = float(cfg.get("shrinkage_strength", 0.1))
    strength = min(max(strength, 0.0), 1.0)
    cov_values = cov.to_numpy(copy=True)

    if shrinkage in {"diagonal", "diag"}:
        target = np.diag(np.diag(cov_values))
    elif shrinkage in {"identity", "eye"}:
        avg_var = float(np.trace(cov_values) / cov_values.shape[0])
        target = np.eye(cov_values.shape[0]) * avg_var
    elif shrinkage in {"constant_correlation", "const_corr"}:
        avg_corr = _average_correlation(cov_values)
        std = np.sqrt(np.diag(cov_values))
        target = np.outer(std, std) * avg_corr
        np.fill_diagonal(target, np.diag(cov_values))
    else:
        return cov

    shrunk = (1.0 - strength) * cov_values + strength * target
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def _average_correlation(cov_values: np.ndarray) -> float:
    std = np.sqrt(np.diag(cov_values))
    if np.any(std == 0):
        return 0.0
    denom = np.outer(std, std)
    corr = cov_values / denom
    n = corr.shape[0]
    if n <= 1:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(corr[mask].mean())

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

    for component in sorted(
        nx.connected_components(subgraph), key=lambda comp: component_score(list(comp))
    ):
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
