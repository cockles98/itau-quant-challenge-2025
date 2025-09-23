from __future__ import annotations

"""Risk control mechanisms and monitoring utilities."""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "apply_caps",
    "apply_turnover_cap",
    "kill_switch",
]


def apply_caps(
    weights: pd.Series,
    max_asset: float = 0.10,
    max_cluster: float = 0.35,
    clusters: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.Series:
    """Apply per-asset and cluster-level caps, renormalising residual capacity."""

    if max_asset <= 0 or max_cluster <= 0:
        raise ValueError("Caps must be positive")

    capped = weights.clip(-max_asset, max_asset)
    residual = 1.0 - capped.sum()

    if clusters:
        capped = _apply_cluster_caps(capped, clusters, max_cluster)
        residual = 1.0 - capped.sum()

    if residual != 0:
        free_assets = capped.index[np.abs(capped) < max_asset]
        if not free_assets.empty:
            adj = residual / len(free_assets)
            capped.loc[free_assets] += adj

    return capped / capped.sum()


def _apply_cluster_caps(
    weights: pd.Series, clusters: Dict[str, Iterable[str]], cap: float
) -> pd.Series:
    adjusted = weights.copy()
    for cluster_assets in clusters.values():
        cluster_assets = [asset for asset in cluster_assets if asset in adjusted.index]
        if not cluster_assets:
            continue
        cluster_weight = adjusted.loc[cluster_assets].sum()
        if abs(cluster_weight) > cap:
            scale = cap / abs(cluster_weight)
            adjusted.loc[cluster_assets] *= scale
    return adjusted


def apply_turnover_cap(
    prev_w: pd.Series, new_w: pd.Series, cap: float = 0.25
) -> pd.Series:
    """Limit one-way turnover, redistributing any excess proportionally."""

    if cap <= 0:
        raise ValueError("cap must be positive")

    aligned_prev = prev_w.reindex(new_w.index).fillna(0.0)
    diff = new_w - aligned_prev
    turnover = diff.abs().sum() / 2.0

    if turnover <= cap:
        return new_w

    scale = cap / turnover
    adjusted = aligned_prev + diff * scale
    return adjusted / adjusted.sum()


def kill_switch(
    equity_curve: pd.Series,
    realized_vol: pd.Series,
    vol_target: float,
    mdd_lookback: int = 90,
    mdd_thres: float = -0.20,
    vol_mult: float = 1.8,
) -> bool:
    """Check drawdown/volatility thresholds to decide if trading should stop."""

    if vol_target <= 0:
        raise ValueError("vol_target must be positive")

    if equity_curve.empty or realized_vol.empty:
        return False

    eq = equity_curve.dropna()
    vol = realized_vol.dropna()
    if eq.empty or vol.empty:
        return False

    recent_eq = eq.tail(mdd_lookback)
    peak = recent_eq.cummax()
    mdd = (recent_eq / peak - 1.0).min()

    recent_vol = vol.iloc[-1]

    return mdd <= mdd_thres or recent_vol > vol_mult * vol_target
