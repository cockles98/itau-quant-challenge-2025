from __future__ import annotations

"""Factor computation helpers for feature pipelines."""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "momentum_12_1",
    "slope_nd",
    "quality_proxy",
    "mix_scores",
]

_DAILY_PER_YEAR = 252
_DAILY_PER_MONTH = 21


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.isnull().all(axis=None):
        raise ValueError("prices contains only NaNs")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    return prices


def _zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    def _z(row: pd.Series) -> pd.Series:
        std = row.std(ddof=0)
        if std == 0 or np.isnan(std):
            return row * 0.0
        return (row - row.mean()) / std

    return df.apply(_z, axis=1)


def _winsorize(df: pd.DataFrame, limits: Tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
    lower_q = df.quantile(limits[0], axis=1)
    upper_q = df.quantile(limits[1], axis=1)
    clipped = np.clip(df.to_numpy(), lower_q.values[:, None], upper_q.values[:, None])
    return pd.DataFrame(clipped, index=df.index, columns=df.columns)


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-1 momentum and cross-sectionally z-score the result."""

    prices = _validate_prices(prices)
    lag_12 = prices.shift(_DAILY_PER_YEAR)
    lag_1 = prices.shift(_DAILY_PER_MONTH)
    momentum = (lag_1 / lag_12) - 1
    momentum = momentum.dropna(how="all")
    return _zscore_cross_section(momentum)


def slope_nd(prices: pd.DataFrame, n: int = 63) -> pd.DataFrame:
    """Estimate the slope of the log-price trend over the past *n* observations."""

    if n <= 1:
        raise ValueError("n must be greater than 1")

    prices = _validate_prices(prices)
    log_prices = np.log(prices)
    x = np.arange(n, dtype=float)
    x_demean = x - x.mean()
    denom = np.dot(x_demean, x_demean)

    def _slope(arr: Iterable[float]) -> float:
        values = np.asarray(arr, dtype=float)
        if np.isnan(values).any():
            return np.nan
        y = values
        y_mean = y.mean()
        numer = np.dot(x_demean, y - y_mean)
        if denom == 0:
            return np.nan
        return numer / denom

    slopes = log_prices.rolling(window=n, min_periods=n).apply(_slope, raw=True)
    slopes = slopes.dropna(how="all")
    return _zscore_cross_section(slopes)


def quality_proxy(prices: pd.DataFrame) -> pd.DataFrame:
    """Proxy for quality using volatility, drawdown and trend persistence."""

    prices = _validate_prices(prices)
    returns = prices.pct_change()
    vol = returns.rolling(window=21, min_periods=21).std(ddof=0)

    window = 126
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = prices / rolling_max - 1.0

    persistence = np.sign(returns).rolling(window=21, min_periods=21).mean()

    combined = (
        (-vol.ffill()) * 0.5
        + (-drawdown.fillna(0)) * 0.3
        + (persistence.fillna(0)) * 0.2
    )

    combined = combined.dropna(how="all")
    return _zscore_cross_section(combined)


def mix_scores(
    regime: pd.Series,
    momentum: pd.DataFrame,
    quality: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    """Blend factor scores with regime awareness and winsorized normalisation."""

    if not (alpha > beta >= gamma):
        raise ValueError("Weights must satisfy alpha > beta >= gamma")

    if not isinstance(regime, pd.Series):
        raise TypeError("regime must be a pandas Series")

    common_index = momentum.index.intersection(quality.index).intersection(regime.index)
    if common_index.empty:
        raise ValueError("No overlapping dates across inputs")

    momentum = momentum.reindex(common_index)
    quality = quality.reindex(common_index)
    regime = regime.reindex(common_index)

    momentum = momentum.ffill()
    quality = quality.ffill()
    regime = regime.ffill().bfill()

    valid = (~momentum.isna().all(axis=1)) & (~quality.isna().all(axis=1)) & (~regime.isna())
    momentum = momentum.loc[valid]
    quality = quality.loc[valid]
    regime = regime.loc[valid]

    regime_values = regime.to_numpy()[:, None]
    combined = regime_values * (alpha * momentum + beta * quality) + (1 - regime_values) * (
        gamma * quality
    )

    combined = pd.DataFrame(combined, index=regime.index, columns=momentum.columns)
    combined = _winsorize(combined)
    return _zscore_cross_section(combined).rename(columns=lambda c: f"mix_{c}")




