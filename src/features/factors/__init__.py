from __future__ import annotations

"""Factor computation helpers for feature pipelines."""

from typing import Any, Dict, Iterable, Tuple

import logging
import os

import numpy as np
import pandas as pd

from .peripherality import peripherality_factor

__all__ = [
    "momentum_12_1",
    "slope_nd",
    "quality_proxy",
    "mix_scores",
    "get_alphas_from_cfg",
    "forward_returns",
    "peripherality_factor",
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


def forward_returns(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Compute forward returns over *horizon* periods for each asset.

    This function is robust to non-positive prices and divisions by zero that
    can appear in broad-market datasets. Any non-finite values in the resulting
    returns are converted to NaN so downstream consumers can safely drop them.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    prices = _validate_prices(prices)
    # Guard against non-positive prices that would induce infinite pct-changes
    safe_prices = prices.where(prices > 0)
    returns = safe_prices.pct_change(periods=horizon, fill_method=None).shift(-horizon)
    # Replace infinities (from division by zero) with NaN for safe downstream use
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns


def get_alphas_from_cfg(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Return (alpha, beta, gamma) priority weights from multiple possible config namespaces.

    It supports:
        - cfg["factors"]["alphas"] = [a, b, c]
        - top-level cfg["alpha"], cfg["beta"], cfg["gamma"]
        - cfg["validation"]["current_params"].{alpha,beta,gamma}
    Fallback is (0.6, 0.3, 0.1).
    """

    try:
        fac = cfg.get("factors", {}) if isinstance(cfg, dict) else {}
        alphas = fac.get("alphas", None)
        if isinstance(alphas, (list, tuple)) and alphas and isinstance(alphas[0], (int, float)):
            a, b, c = (list(alphas) + [0.3, 0.1])[:3]
            return float(a), float(b), float(c)
        # top-level direct values
        a = cfg.get("alpha")
        b = cfg.get("beta")
        c = cfg.get("gamma")
        if all(isinstance(x, (int, float)) for x in (a, b, c)):
            return float(a), float(b), float(c)
        # validation.current_params fallback
        vp = (cfg.get("validation", {}) or {}).get("current_params", {}) if isinstance(cfg, dict) else {}
        if all(k in vp for k in ("alpha", "beta", "gamma")):
            return float(vp["alpha"]), float(vp["beta"]), float(vp["gamma"])
    except Exception:
        pass
    return 0.6, 0.3, 0.1


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

    combined = (-vol.ffill()) * 0.5 + (-drawdown.fillna(0)) * 0.3 + (persistence.fillna(0)) * 0.2

    combined = combined.dropna(how="all")
    return _zscore_cross_section(combined)


def mix_scores(
    regime: pd.Series,
    momentum: pd.DataFrame,
    quality: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    *,
    peripherality: pd.DataFrame | None = None,
    use_peripherality: bool | None = None,
    delta: float | None = None,
    regime_gain: float | None = None,
    regime_mode: str | None = None,
) -> pd.DataFrame:
    """Blend factor scores with regime awareness and optional peripherality tilt."""

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

    periph_matrix: pd.DataFrame | None = None
    periph_flag = bool(use_peripherality)
    if use_peripherality is None:
        periph_flag = peripherality is not None and delta is not None
    if periph_flag:
        if peripherality is None:
            raise ValueError("peripherality scores must be provided when use_peripherality is True")
        if delta is None:
            raise ValueError("delta weight is required when use_peripherality is True")
        periph_matrix = peripherality.reindex(common_index)
        periph_matrix = periph_matrix.ffill()

    valid = (~momentum.isna().all(axis=1)) & (~quality.isna().all(axis=1)) & (~regime.isna())

    if periph_matrix is not None:
        valid &= ~periph_matrix.isna().all(axis=1)

    momentum = momentum.loc[valid]
    quality = quality.loc[valid]
    regime = regime.loc[valid]
    if periph_matrix is not None:
        periph_matrix = periph_matrix.loc[valid]

    r = np.clip(regime.to_numpy()[:, None], 0.0, 1.0)

    env_gain = os.getenv("REGIME_GAIN")
    gain_source = regime_gain if regime_gain is not None else env_gain
    try:
        gain = float(gain_source) if gain_source is not None else 1.0
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("Invalid regime_gain '%s'; falling back to 1.0", gain_source)
        gain = 1.0
    mode_raw = regime_mode if regime_mode is not None else os.getenv("REGIME_MODE", "tanh")
    mode = str(mode_raw).lower()
    if mode not in {"tanh", "linear", "power"}:
        logging.getLogger(__name__).warning("Invalid regime_mode '%s'; falling back to 'tanh'", mode_raw)
        mode = "tanh"
    if mode == "tanh":
        z = (r - 0.5) * 2.0
        r_eff = 0.5 * (1.0 + np.tanh(gain * z))
    elif mode == "power":
        r_eff = np.power(r, max(1e-6, gain))
    else:
        r_eff = r

    w_mom = alpha * r_eff
    w_qual = beta * r_eff + gamma * (1.0 - r_eff)
    combined = w_mom * momentum + w_qual * quality

    if periph_matrix is not None:
        combined = combined + float(delta) * periph_matrix

    combined = pd.DataFrame(combined, index=regime.index, columns=momentum.columns)
    combined = _winsorize(combined)
    return combined.rename(columns=lambda c: f"mix_{c}")

