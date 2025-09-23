from __future__ import annotations

"""Position sizing rules and allocation utilities."""

import numpy as np
import pandas as pd

__all__ = [
    "atr",
    "atr_risk_normalize",
    "scale_to_vol",
]


def atr(prices: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Average True Range using simple high-low-close approximation."""

    if n <= 1:
        raise ValueError("n must be greater than 1")
    if prices.isnull().all(axis=None):
        raise ValueError("prices contain only NaNs")

    returns = prices.pct_change()
    true_range = returns.abs()
    atr_values = true_range.rolling(window=n, min_periods=n).mean()
    return atr_values.dropna(how="all")


def atr_risk_normalize(weights: pd.Series, atr_df: pd.DataFrame) -> pd.Series:
    """Scale weights inversely proportional to ATR per asset."""

    latest_atr = atr_df.iloc[-1]
    aligned = weights.reindex(latest_atr.index).dropna()
    latest_atr = latest_atr.loc[aligned.index]
    inv_risk = 1.0 / latest_atr.replace(0, np.nan)
    inv_risk = inv_risk.fillna(0.0)

    scaled = aligned * inv_risk
    total = scaled.abs().sum()
    if total == 0:
        raise ValueError("ATR normalisation resulted in zero total weight")
    return scaled / total


def scale_to_vol(
    weights: pd.Series,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    window: int = 60,
) -> pd.Series:
    """Rescale weights to hit a target realised volatility over a rolling window."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")

    aligned_returns = returns.reindex(columns=weights.index).dropna(how="all")
    latest_returns = aligned_returns.tail(window)
    cov = latest_returns.cov()
    active_assets = weights.index.intersection(cov.index)
    if active_assets.empty:
        raise ValueError("No overlapping assets between weights and returns")

    active_weights = weights.loc[active_assets]
    cov = cov.loc[active_assets, active_assets]

    portfolio_var = float(active_weights @ cov @ active_weights)
    if portfolio_var <= 0:
        raise ValueError("Portfolio variance is non-positive")

    current_vol = np.sqrt(portfolio_var)
    scaling = target_vol / current_vol
    return weights * scaling
