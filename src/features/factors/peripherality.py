from __future__ import annotations

"""Peripherality-based factor utilities."""

import numpy as np
import pandas as pd

__all__ = ["peripherality_factor"]


def peripherality_factor(
    per_node_centrality: pd.Series,
    *,
    clip: float = 3.0,
) -> pd.Series:
    """Convert node centrality scores into a peripheral factor tilt.

    Parameters
    ----------
    per_node_centrality : pd.Series
        Centrality values indexed by asset identifier. Higher centrality implies
        more connectivity; the factor inverts this so that lower centrality
        (i.e. more peripheral) assets receive higher scores.
    clip : float, default 3.0
        Absolute clip applied to the z-scored output to guard against outliers.

    Returns
    -------
    pd.Series
        Z-scored, clipped peripherality factor aligned to the input index.
    """

    if not isinstance(per_node_centrality, pd.Series):
        raise TypeError("per_node_centrality must be a pandas Series.")

    if per_node_centrality.empty:
        return pd.Series(dtype=float)

    values = per_node_centrality.astype(float)
    inverted = -values  # lower centrality -> higher peripherality
    mean = float(inverted.mean())
    std = float(inverted.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=values.index, dtype=float)

    z_scores = (inverted - mean) / std
    clipped = np.clip(z_scores, -abs(clip), abs(clip))
    return pd.Series(clipped, index=values.index, dtype=float, name="peripherality")

