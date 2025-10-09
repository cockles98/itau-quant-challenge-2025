from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from ..tda.ph_turbulence import PHTurbulenceTransformer
except ImportError:  # pragma: no cover - optional dependency (giotto-tda)
    PHTurbulenceTransformer = None


def compute_ph_regime_index(
    returns: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.Series:
    """Compute a PH-based turbulence regime index."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas.DataFrame.")
    if returns.empty:
        raise ValueError("returns must contain at least one row.")
    if PHTurbulenceTransformer is None:
        raise ImportError(
            "PHTurbulenceTransformer requires the 'giotto-tda' package. "
            "Install the optional dependency to enable regime computation."
        )

    tda_cfg = cfg.get("tda_ph", {})
    enabled = bool(tda_cfg.get("enabled", True))
    span = int(tda_cfg.get("smooth_span", 10))
    lookback = int(tda_cfg.get("zscore_lookback", 250))
    alert_sigma = float(tda_cfg.get("alert_sigma", 1.0))
    riskoff_sigma = float(tda_cfg.get("riskoff_sigma", 2.0))
    if riskoff_sigma < alert_sigma:
        raise ValueError("riskoff_sigma must be greater than or equal to alert_sigma.")

    base_series = _compute_base_series(returns, tda_cfg) if enabled else None

    if base_series is None:
        index = pd.Index(returns.index, name=returns.index.name)
        regime = pd.Series(0.0, index=index, name="ph_regime")
        alert_flag = pd.Series(False, index=index, name="is_alert")
        riskoff_flag = pd.Series(False, index=index, name="is_riskoff")
        regime.attrs["is_alert"] = alert_flag
        regime.attrs["is_riskoff"] = riskoff_flag
        return regime

    smoothed = base_series.ewm(span=span, adjust=False).mean()
    rolling_mean = smoothed.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = smoothed.rolling(window=lookback, min_periods=lookback).std(ddof=0)
    z_values = (smoothed - rolling_mean) / rolling_std

    # Avoid division by zero and NaN propagation.
    z_values = z_values.replace([np.inf, -np.inf], np.nan)

    denom = max(riskoff_sigma - alert_sigma, 1e-9)
    regime_value = ((z_values - alert_sigma) / denom).clip(lower=0.0, upper=1.0)
    regime_value = regime_value.fillna(0.0)
    regime_value.name = "ph_regime"

    alert_flag = (z_values >= alert_sigma).fillna(False)
    riskoff_flag = (z_values >= riskoff_sigma).fillna(False)

    regime_value.attrs["is_alert"] = alert_flag.rename("is_alert")
    regime_value.attrs["is_riskoff"] = riskoff_flag.rename("is_riskoff")
    regime_value.attrs["zscore_series"] = z_values.rename("ph_regime_zscore")

    return regime_value


def _compute_base_series(
    returns: pd.DataFrame,
    tda_cfg: Mapping[str, Any],
) -> pd.Series | None:
    if PHTurbulenceTransformer is None:
        raise ImportError(
            "PHTurbulenceTransformer requires the 'giotto-tda' package. "
            "Install the optional dependency to enable regime computation."
        )
    transformer = PHTurbulenceTransformer(
        window=int(tda_cfg.get("window", 50)),
        homology_dimensions=(int(tda_cfg.get("homology_dim", 1)),),
        norm=str(tda_cfg.get("norm", "l2")),
        compute_entropy=False,
        compute_amplitude=False,
        compute_wasserstein=False,
    )
    try:
        transformer.fit(returns)
        series = transformer.transform(returns)
    except ValueError:
        index = pd.Index(returns.index, name=returns.index.name)
        series = pd.Series(np.nan, index=index, name="ph_turbulence")
    return series
