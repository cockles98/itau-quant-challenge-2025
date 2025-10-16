from __future__ import annotations

"""Peripherality-biased weighting adjustments."""

import numpy as np
import pandas as pd


def apply_periphery_bias(
    weights: pd.Series,
    centrality: pd.Series | None,
    lam: float,
) -> pd.Series:
    """Apply a centrality penalty to HRP weights.

    Parameters
    ----------
    weights : pd.Series
        Base weights (e.g. HRP output), assumed to sum to 1.
    centrality : pd.Series or None
        Asset centrality scores aligned by index. Higher values mean more central
        (i.e., less peripheral). If ``None`` or empty the weights are returned
        unchanged.
    lam : float
        Non-negative penalty strength. ``0`` disables the adjustment.

    Returns
    -------
    pd.Series
        Reweighted series summing to 1.
    """
    if lam <= 0 or centrality is None or centrality.empty:
        total = weights.sum()
        return weights if total == 1.0 else weights / total if total else weights

    aligned_centrality = centrality.reindex(weights.index)
    if aligned_centrality.isna().all():
        total = weights.sum()
        return weights if total == 1.0 else weights / total if total else weights

    # Replace missing centrality with the average observed to avoid biasing a few assets.
    mean_centrality = float(aligned_centrality.mean(skipna=True)) if not aligned_centrality.isna().all() else 0.0
    aligned_centrality = aligned_centrality.fillna(mean_centrality).clip(lower=0.0)

    # Exponential penalty preserves positivity and handles any scale of centrality.
    penalties = np.exp(-lam * aligned_centrality.to_numpy(dtype=float))
    adjusted = weights.to_numpy(dtype=float) * penalties
    adjusted = np.clip(adjusted, a_min=0.0, a_max=None)

    total = adjusted.sum()
    if total <= 0:
        total = weights.sum()
        adjusted = weights.to_numpy(dtype=float)

    adjusted_series = pd.Series(adjusted / total, index=weights.index, dtype=float)
    return adjusted_series
