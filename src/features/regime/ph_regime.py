from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import numpy as np

try:
    from ..tda.ph_turbulence import PHTurbulenceTransformer
except ImportError:  # pragma: no cover - optional dependency (giotto-tda)
    PHTurbulenceTransformer = None


def _cache_path(
    returns: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> Optional[Path]:
    """Return cache path for the PH regime result."""
    if not isinstance(cfg, Mapping):
        return None
    paths_cfg = cfg.get("paths", {}) or {}
    artifacts_root = Path(paths_cfg.get("artifacts", "./artifacts"))
    cache_dir = artifacts_root / "cache" / "regime"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    tda_cfg = cfg.get("tda_ph", {}) or {}
    payload = {
        "shape": returns.shape,
        "columns": [str(col) for col in returns.columns],
        "index_start": str(pd.Timestamp(returns.index.min())),
        "index_end": str(pd.Timestamp(returns.index.max())),
        "tda_ph": tda_cfg,
    }
    try:
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        data_hash = pd.util.hash_pandas_object(returns, index=True).values.tobytes()
    except Exception:
        return None

    digest = hashlib.md5()
    digest.update(payload_bytes)
    digest.update(data_hash)
    filename = f"ph_regime_{digest.hexdigest()}.parquet"
    return cache_dir / filename


def _load_cached_regime(path: Path) -> Optional[pd.Series]:
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, ImportError):
        return None
    if "regime" not in frame.columns:
        return None
    regime = frame["regime"]
    regime.name = "ph_regime"
    attrs = {}
    if "is_alert" in frame.columns:
        attrs["is_alert"] = frame["is_alert"].astype(bool).rename("is_alert")
    if "is_riskoff" in frame.columns:
        attrs["is_riskoff"] = frame["is_riskoff"].astype(bool).rename("is_riskoff")
    if "zscore" in frame.columns:
        attrs["zscore_series"] = frame["zscore"].rename("ph_regime_zscore")
    for key, value in attrs.items():
        regime.attrs[key] = value
    return regime


def _store_cached_regime(path: Path, regime: pd.Series) -> None:
    alert_series = regime.attrs.get("is_alert")
    riskoff_series = regime.attrs.get("is_riskoff")
    zscore_series = regime.attrs.get("zscore_series")
    frame = pd.DataFrame({"regime": regime})
    if isinstance(alert_series, pd.Series):
        frame["is_alert"] = alert_series.reindex(regime.index).astype(bool)
    if isinstance(riskoff_series, pd.Series):
        frame["is_riskoff"] = riskoff_series.reindex(regime.index).astype(bool)
    if isinstance(zscore_series, pd.Series):
        frame["zscore"] = zscore_series.reindex(regime.index)
    try:
        frame.to_parquet(path, compression="snappy")
    except (OSError, ValueError, ImportError):
        return


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

    cache_path = _cache_path(returns, cfg)
    if cache_path and cache_path.exists():
        cached = _load_cached_regime(cache_path)
        if cached is not None:
            return cached

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
        if cache_path:
            _store_cached_regime(cache_path, regime)
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

    if cache_path:
        _store_cached_regime(cache_path, regime_value)
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
