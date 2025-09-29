from __future__ import annotations

"""Factor computation helpers for feature pipelines."""

from typing import Iterable, Tuple, Any, Dict

import numpy as np
import pandas as pd

__all__ = [
    "momentum_12_1",
    "slope_nd",
    "quality_proxy",
    "mix_scores",
    # novo helper p/ engine:
    "get_alphas_from_cfg",
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

# --------------------------------------
# NOVO: leitura robusta de alphas (α,β,γ)
# --------------------------------------
def get_alphas_from_cfg(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Retorna (alpha, beta, gamma) a partir de:
    - cfg["factors"]["alphas"] = [a,b,c]
    - topo do YAML: cfg["alpha"], cfg["beta"], cfg["gamma"]
    - (opcional) cfg["validation"]["current_params"].{alpha,beta,gamma}
    Fallback: (0.6, 0.3, 0.1)
    """
    try:
        fac = cfg.get("factors", {}) if isinstance(cfg, dict) else {}
        alphas = fac.get("alphas", None)
        if isinstance(alphas, (list, tuple)) and alphas and isinstance(alphas[0], (int, float)):
            a, b, c = (list(alphas) + [0.3, 0.1])[:3]
            return float(a), float(b), float(c)
        # topo direto
        a = cfg.get("alpha"); b = cfg.get("beta"); c = cfg.get("gamma")
        if all(isinstance(x, (int, float)) for x in (a, b, c)):
            return float(a), float(b), float(c)
        # validation.current_params (se você utilizar depois)
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


def _winsorize(
    df: pd.DataFrame, limits: Tuple[float, float] = (0.05, 0.95)
) -> pd.DataFrame:
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

    valid = (
        (~momentum.isna().all(axis=1))
        & (~quality.isna().all(axis=1))
        & (~regime.isna())
    )
    momentum = momentum.loc[valid]
    quality = quality.loc[valid]
    regime = regime.loc[valid]

    # regime_values = regime.to_numpy()[:, None]
    # # combined = regime_values * (alpha * momentum + beta * quality) + (
    # #     1 - regime_values
    # # ) * (gamma * quality)
    # w_mom = alpha * regime_values                 # sobe momentum em regime alto
    # w_qual = beta * regime_values + gamma*(1-regime_values)  # puxa quality quando regime é baixo
    import os
    r = np.clip(regime.to_numpy()[:, None], 0.0, 1.0)
    # ===== Realce não-linear controlado por env =====
    # REGIME_GAIN>1 "puxa" r para os extremos; <1 suaviza.
    gain = float(os.getenv("REGIME_GAIN", "1.0"))
    mode = os.getenv("REGIME_MODE", "tanh")  # {"tanh","linear","power"}
    if mode == "tanh":
        # mapeia r∈[0,1] → r'∈[0,1] com S-curve controlada por 'gain'
        z = (r - 0.5) * 2.0
        r_eff = 0.5 * (1.0 + np.tanh(gain * z))
    elif mode == "power":
        # r' = r^gain (mantém [0,1]); gain>1 acentua baixos/altos
        r_eff = np.power(r, max(1e-6, gain))
    else:
        r_eff = r
    # ===============================================
    w_mom = alpha * r_eff
    w_qual = beta * r_eff + gamma * (1.0 - r_eff)
    combined = w_mom * momentum + w_qual * quality

    # NÃO padronizar de novo – isso anulava o efeito do 'regime'.
    # combined = pd.DataFrame(combined, index=regime.index, columns=momentum.columns)
    # combined = _winsorize(combined)
    # return combined.rename(columns=lambda c: f"mix_{c}")
    combined = pd.DataFrame(combined, index=regime.index, columns=momentum.columns)
    combined = _winsorize(combined)
    return combined.rename(columns=lambda c: f"mix_{c}")