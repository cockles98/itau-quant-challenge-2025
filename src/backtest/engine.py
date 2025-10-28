from __future__ import annotations

"""Core backtesting engine scaffolding."""

import os
import json
import hashlib
from dataclasses import dataclass
import logging
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import dump, load, Parallel, delayed

from backtest.execution import execute_trade
from dataio.loaders import get_adv, get_panel, select_universe
from features import (
    mix_scores,
    momentum_12_1,
    quality_proxy,
    get_alphas_from_cfg,
    forward_returns,
    peripherality_factor,
    RegimeAwareMapper,
    compute_ph_regime_index,
)
from portfolio import (
    expected_sharpe_tilt,
    hrp_weights_from_order,
    rolling_cov,
    topo_seriation_from_graph,
)
from risk import atr, atr_risk_normalize, scale_to_vol
from risk import guards
from risk import risk_controls
from portfolio.weighting import apply_periphery_bias
from models.meta_blend import run_meta_blend
from metrics import avg_time_under_water, max_time_under_water

logger = logging.getLogger(__name__)


def _factor_cache_dir(paths_cfg: Dict[str, object]) -> Path:
    artifacts_root = Path(paths_cfg.get("artifacts", "./artifacts"))
    cache_dir = artifacts_root / "cache" / "factors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _data_signature(paths_cfg: Dict[str, object]) -> str:
    data_root = Path(paths_cfg.get("data", "./data"))
    try:
        csv_files = list(data_root.glob("*.csv"))
    except OSError:
        return "na"
    if not csv_files:
        return "na"
    mtimes: List[int] = []
    for path in csv_files:
        try:
            mtimes.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    if not mtimes:
        return "na"
    return f"{max(mtimes)}_{len(csv_files)}"


def _cov_cache_dir(paths_cfg: Dict[str, object]) -> Path:
    artifacts_root = Path(paths_cfg.get("artifacts", "./artifacts"))
    cache_dir = artifacts_root / "cache" / "cov"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _universe_cache_dir(paths_cfg: Dict[str, object]) -> Path:
    artifacts_root = Path(paths_cfg.get("artifacts", "./artifacts"))
    cache_dir = artifacts_root / "cache" / "universe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _mapper_cache_dir(paths_cfg: Dict[str, object]) -> Path:
    artifacts_root = Path(paths_cfg.get("artifacts", "./artifacts"))
    cache_dir = artifacts_root / "cache" / "mapper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _mapper_cache_path(
    cache_dir: Path,
    date: pd.Timestamp,
    lookback_returns: pd.DataFrame,
    mapper_params: Dict[str, object],
    regime_value: float,
    lookback: int,
    min_history: int,
) -> Optional[Path]:
    try:
        payload = {
            "date": str(pd.Timestamp(date)),
            "shape": lookback_returns.shape,
            "columns": [str(col) for col in lookback_returns.columns],
            "lookback": int(lookback),
            "min_history": int(min_history),
            "regime": float(regime_value),
            "mapper_params": {str(k): mapper_params.get(k) for k in sorted(mapper_params.keys())},
        }
        digest = hashlib.md5()
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        normalized = (
            lookback_returns.sort_index(axis=0).sort_index(axis=1)
        )
        data_hash = pd.util.hash_pandas_object(
            normalized, index=True
        ).values.tobytes()
        digest.update(data_hash)
        filename = f"mapper_{digest.hexdigest()}.pkl"
    except Exception:
        return None
    return cache_dir / filename


def _load_mapper_cache(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = load(path)
    except (OSError, ValueError, EOFError):
        return None
    if not isinstance(payload, dict):
        return None

    periphery_raw = payload.get("peripherality")
    periphery_series: Optional[pd.Series]
    if isinstance(periphery_raw, (list, tuple)):
        try:
            periphery_series = pd.Series(
                {str(asset): float(value) for asset, value in periphery_raw},
                dtype=float,
            )
        except Exception:
            periphery_series = None
    elif isinstance(periphery_raw, pd.Series):
        periphery_series = periphery_raw.astype(float)
    else:
        periphery_series = None

    centrality = payload.get("centrality")
    if isinstance(centrality, dict):
        centrality = {str(key): float(value) for key, value in centrality.items()}
    else:
        centrality = {}

    metrics_summary = payload.get("metrics_summary")
    if isinstance(metrics_summary, dict):
        metrics_summary = {str(key): float(value) for key, value in metrics_summary.items()}
    else:
        metrics_summary = {}

    return {
        "peripherality": periphery_series,
        "centrality": centrality,
        "payload": payload.get("payload") or {},
        "artifacts": payload.get("artifacts"),
        "metrics_summary": metrics_summary,
    }


def _store_mapper_cache(
    path: Path,
    periphery: pd.Series,
    centrality: Dict[str, float],
    payload: Dict[str, object],
    metrics_summary: Dict[str, float],
    artifacts: Optional[Dict[str, str]] = None,
) -> None:
    record = {
        "peripherality": [
            (str(idx), float(value)) for idx, value in periphery.items()
        ],
        "centrality": {str(key): float(value) for key, value in (centrality or {}).items()},
        "payload": payload,
        "metrics_summary": {str(key): float(value) for key, value in (metrics_summary or {}).items()},
        "artifacts": artifacts,
    }
    try:
        dump(record, path)
    except (OSError, ValueError):
        return


def _stable_hash(payload: str) -> str:
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _lerp(low: float, high: float, weight: float) -> float:
    weight = float(np.clip(weight, 0.0, 1.0))
    return low + (high - low) * weight


def _regime_target_vol(
    base_target: float,
    regime_value: float,
    cfg: Optional[Dict[str, float]] = None,
) -> tuple[float, float]:
    cfg = cfg or {}
    scale_low = float(
        cfg.get(
            "scale_low",
            cfg.get("low", cfg.get("min", 0.8)),
        )
    )
    scale_high = float(
        cfg.get(
            "scale_high",
            cfg.get("high", cfg.get("max", 1.2)),
        )
    )
    scale_low = max(scale_low, 0.0)
    scale_high = max(scale_high, scale_low or 0.0)
    scale = max(0.0, _lerp(scale_low, scale_high, regime_value))
    target = max(1e-6, base_target * (scale if scale > 0 else 1.0))
    return target, scale


def _regime_gross_target(
    regime_value: float,
    cfg: Optional[Dict[str, float]] = None,
) -> float:
    if not isinstance(cfg, dict):
        return 1.0
    low = float(cfg.get("low", cfg.get("min", 1.0)))
    high = float(cfg.get("high", cfg.get("max", 1.0)))
    low = max(low, 0.0)
    high = max(high, low)
    return float(np.clip(_lerp(low, high, regime_value), 0.0, 1.0))


def _resolve_bounds(
    base: float,
    cfg: Optional[Dict[str, float]] = None,
    *,
    minimum: float = 1e-6,
    default_delta: float = 0.0,
) -> tuple[float, float]:
    low = high = None
    delta = default_delta
    if isinstance(cfg, dict):
        if "delta" in cfg:
            delta = float(cfg["delta"])
        low = cfg.get("low", cfg.get("min"))
        high = cfg.get("high", cfg.get("max"))
    elif isinstance(cfg, (int, float)):
        delta = float(cfg)
    if low is None and high is None and delta:
        low = base * max(0.0, 1.0 - delta)
        high = base * (1.0 + delta)
    if low is None:
        low = base
    if high is None:
        high = base
    low = max(float(low), minimum)
    high = max(float(high), low)
    return low, high


def _generate_rolling_windows(
    unique_dates: pd.Index, in_sample_len: int, oos_len: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    if in_sample_len <= 0 or oos_len <= 0:
        return windows
    total = len(unique_dates)
    current_idx = 0
    while current_idx + in_sample_len + oos_len <= total:
        is_start = unique_dates[current_idx]
        is_end = unique_dates[current_idx + in_sample_len - 1]
        oos_start = unique_dates[current_idx + in_sample_len]
        oos_end = unique_dates[current_idx + in_sample_len + oos_len - 1]
        windows.append((is_start, is_end, oos_start, oos_end))
        current_idx += oos_len
    return windows


def _run_backtest_rolling(
    cfg: Dict,
    panel: pd.DataFrame,
    rolling_cfg: Dict[str, object],
) -> Dict[str, object]:
    in_sample_len = int(
        rolling_cfg.get("in_sample_days", rolling_cfg.get("train_days", 252 * 2))
    )
    oos_len = int(
        rolling_cfg.get("out_of_sample_days", rolling_cfg.get("test_days", 126))
    )
    if in_sample_len <= 0 or oos_len <= 0:
        logger.warning(
            "Invalid rolling-training configuration detected; falling back to single backtest run."
        )
        cfg_fallback = deepcopy(cfg)
        cfg_fallback.setdefault("backtest", {}).setdefault(
            "rolling_training", {}
        )["enabled"] = False
        return run_backtest(cfg_fallback, panel=panel)

    panel_sorted = panel.sort_index()
    date_level = (
        "date" if "date" in panel_sorted.index.names else panel_sorted.index.names[0]
    )
    unique_dates = (
        panel_sorted.index.get_level_values(date_level).unique().sort_values()
    )
    windows = _generate_rolling_windows(unique_dates, in_sample_len, oos_len)
    if not windows:
        logger.warning(
            "Rolling-training enabled but no valid windows were generated; running single backtest instead."
        )
        cfg_fallback = deepcopy(cfg)
        cfg_fallback.setdefault("backtest", {}).setdefault(
            "rolling_training", {}
        )["enabled"] = False
        return run_backtest(cfg_fallback, panel=panel)

    idx_dates = panel_sorted.index.get_level_values(date_level)
    combined_returns: List[Tuple[pd.Timestamp, float]] = []
    combined_equity: List[Tuple[pd.Timestamp, float]] = []
    positions_frames: List[pd.DataFrame] = []
    trades_records: List[Dict[str, object]] = []
    windows_meta: List[Dict[str, object]] = []
    current_equity = 1.0
    final_weights: Optional[pd.Series] = None

    for window_id, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        mask = (idx_dates >= is_start) & (idx_dates <= oos_end)
        panel_slice = panel_sorted.loc[mask]
        if panel_slice.empty:
            continue

        cfg_window = deepcopy(cfg)
        cfg_window.setdefault("dates", {})
        cfg_window["dates"]["start"] = is_start.isoformat()
        cfg_window["dates"]["end"] = oos_end.isoformat()
        cfg_window.setdefault("backtest", {}).setdefault(
            "rolling_training", {}
        )["enabled"] = False

        window_result = run_backtest(cfg_window, panel=panel_slice)
        equity_curve_full = window_result.get("equity_curve")
        if equity_curve_full is None or equity_curve_full.empty:
            continue

        returns_full = equity_curve_full.pct_change(fill_method=None).fillna(0.0)
        returns_oos = returns_full.loc[oos_start:oos_end]
        if returns_oos.empty:
            continue

        for date, ret in returns_oos.items():
            current_equity *= 1.0 + float(ret)
            combined_returns.append((date, float(ret)))
            combined_equity.append((date, current_equity))

        positions = window_result.get("daily_positions")
        if isinstance(positions, pd.DataFrame) and not positions.empty:
            positions_frames.append(positions.loc[oos_start:oos_end])

        window_trades = window_result.get("trades")
        if window_trades is None:
            trades_iter = []
        elif isinstance(window_trades, pd.DataFrame):
            trades_iter = window_trades.to_dict("records")
        else:
            trades_iter = list(window_trades)
        for trade in trades_iter:
            trade_date = trade.get("date")
            if trade_date is None:
                continue
            ts_date = pd.Timestamp(trade_date)
            if oos_start <= ts_date <= oos_end:
                trades_records.append(trade)

        final_weights = window_result.get("weights", final_weights)
        windows_meta.append(
            {
                "window": window_id,
                "is_start": is_start.isoformat(),
                "is_end": is_end.isoformat(),
                "oos_start": oos_start.isoformat(),
                "oos_end": oos_end.isoformat(),
                "kpis": window_result.get("kpis", {}),
                "meta": window_result.get("meta", {}),
            }
        )

    if not combined_equity:
        logger.warning(
            "Rolling-training produced no equity data; falling back to single backtest run."
        )
        cfg_fallback = deepcopy(cfg)
        cfg_fallback.setdefault("backtest", {}).setdefault(
            "rolling_training", {}
        )["enabled"] = False
        return run_backtest(cfg_fallback, panel=panel)

    combined_equity_series = (
        pd.Series({date: value for date, value in combined_equity}, dtype=float)
        .sort_index()
        .rename("equity")
    )
    combined_returns_series = pd.Series(
        {date: value for date, value in combined_returns}, dtype=float
    ).sort_index()

    kpis = _compute_kpis(combined_returns_series, combined_equity_series)

    if positions_frames:
        daily_positions = pd.concat(positions_frames, axis=0).sort_index()
    else:
        daily_positions = pd.DataFrame(dtype=float)

    trades_records = sorted(
        trades_records,
        key=lambda record: pd.Timestamp(record.get("date"))
        if record.get("date") is not None
        else pd.Timestamp.min,
    )

    rolling_meta = {
        "enabled": True,
        "in_sample_days": in_sample_len,
        "out_of_sample_days": oos_len,
        "windows": windows_meta,
    }

    meta = {
        "start": windows[0][0].isoformat(),
        "end": windows[-1][3].isoformat(),
        "rolling_training": rolling_meta,
    }

    return {
        "equity_curve": combined_equity_series,
        "daily_positions": daily_positions,
        "trades": trades_records,
        "weights": final_weights,
        "kpis": kpis,
        "meta": meta,
    }


__all__ = ["run_backtest", "softmax_with_temperature"]


@dataclass
class BacktestState:
    equity: float
    current_weights: pd.Series
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    vol_series: pd.Series
    weights_history: Dict[pd.Timestamp, pd.Series]
    trades: List[Dict[str, float]]
    last_kill_date: Optional[pd.Timestamp] = None


def softmax_with_temperature(values: pd.Series, temperature: float) -> pd.Series:
    """Compute a temperature-controlled softmax with numerical safeguards."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if values.empty:
        return pd.Series(dtype=float)

    scaled = values.to_numpy(dtype=float) / max(temperature, 1e-12)
    scaled -= np.nanmax(scaled)
    exps = np.exp(scaled)
    denom = np.nansum(exps)
    if not np.isfinite(denom) or denom <= 0:
        n = len(values)
        return pd.Series(1.0 / n, index=values.index, dtype=float)

    weights = exps / denom
    return pd.Series(weights, index=values.index, dtype=float)



def run_backtest(cfg: Dict, panel: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Run a deterministic monthly-rebalanced HRP backtest."""

    seed = int(cfg.get("seeds", 42))
    np.random.seed(seed)
    pd.options.mode.copy_on_write = True
    threads = max(1, (os.cpu_count() or 1))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))

    dates_cfg = cfg.get("dates", {})
    paths_cfg = cfg.get("paths", {}) or {}
    start = pd.Timestamp(dates_cfg.get("start"))
    end = pd.Timestamp(dates_cfg.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Missing start/end dates in configuration")

    logger.info("Loading market data from %s to %s", start.date(), end.date())
    if panel is not None:
        panel_data = panel.sort_index()
    else:
        panel_data = get_panel(start, end)

    if not isinstance(panel_data.index, pd.MultiIndex):
        raise ValueError("panel must have a MultiIndex with a date level")

    date_level = (
        "date" if "date" in panel_data.index.names else panel_data.index.names[0]
    )
    date_index = panel_data.index.get_level_values(date_level)
    mask = (date_index >= start) & (date_index <= end)
    panel = panel_data.loc[mask]
    if panel.empty:
        raise ValueError("Panel slice is empty for requested date range")

    rolling_cfg = (cfg.get("backtest", {}) or {}).get("rolling_training", {}) or {}
    if rolling_cfg.get("enabled"):
        return _run_backtest_rolling(cfg, panel, rolling_cfg)

    prices = panel["close"].unstack("asset").sort_index()
    prices_full = prices

    cache_dir = _factor_cache_dir(paths_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_ds_cache_dir = cache_dir / "meta_blend"
    meta_ds_cache_dir.mkdir(parents=True, exist_ok=True)
    data_sig = _data_signature(paths_cfg)
    cache_tag_full = f"{start:%Y%m%d}_{end:%Y%m%d}_{len(prices_full)}x{len(prices_full.columns)}_{data_sig}"

    returns = prices.pct_change(fill_method=None)
    # Replace non-finite returns (e.g., division by zero when prior price is 0)
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logger.info("Computing liquidity metrics and volatility estimates")
    adv_series = get_adv(panel)

    # Pre-compute the trading universe once and reuse it (also for meta-blend filtering)
    universe_cfg = cfg.get("universe", {})
    logger.info("Selecting tradable universe with hysteresis")
    universe_dir = _universe_cache_dir(paths_cfg)
    universe_payload = {
        "top_n": int(universe_cfg.get("top_n", 20)),
        "adv_min": float(universe_cfg.get("adv_min", 0)),
        "price_min": float(universe_cfg.get("price_min", 0)),
        "age_min": int(universe_cfg.get("age_min", 20)),
        "hysteresis": int(universe_cfg.get("hysteresis_rebalances", 2)),
        "calendar": str(universe_cfg.get("calendar", "BM")),
    }
    universe_hash = _stable_hash(json.dumps(universe_payload, sort_keys=True))
    universe_cache_path = universe_dir / f"universe_{cache_tag_full}_{universe_hash}.json"
    universe_map: Dict[pd.Timestamp, List[str]]
    if universe_cache_path.exists():
        try:
            raw = json.loads(universe_cache_path.read_text(encoding="utf-8"))
            universe_map = {
                pd.Timestamp(key): list(value) for key, value in raw.items()
            }
        except (OSError, ValueError, json.JSONDecodeError):
            universe_map = select_universe(
                panel,
                top_n=universe_payload["top_n"],
                adv_min=universe_payload["adv_min"],
                price_min=universe_payload["price_min"],
                age_min=universe_payload["age_min"],
                hysteresis_rebalances=universe_payload["hysteresis"],
                calendar=universe_payload["calendar"],
                adv_series=adv_series,
            )
    else:
        universe_map = select_universe(
            panel,
            top_n=universe_payload["top_n"],
            adv_min=universe_payload["adv_min"],
            price_min=universe_payload["price_min"],
            age_min=universe_payload["age_min"],
            hysteresis_rebalances=universe_payload["hysteresis"],
            calendar=universe_payload["calendar"],
            adv_series=adv_series,
        )
        try:
            serialised = {dt.isoformat(): assets for dt, assets in universe_map.items()}
            universe_cache_path.write_text(json.dumps(serialised), encoding="utf-8")
        except OSError:
            pass

    eligible_assets = (
        sorted({asset for asset_list in universe_map.values() for asset in asset_list})
        if universe_map
        else list(prices.columns)
    )

    if eligible_assets:
        prices = prices.reindex(columns=eligible_assets)
        returns = returns.reindex(columns=eligible_assets)
        adv_series = adv_series.loc[
            adv_series.index.get_level_values("asset").isin(eligible_assets)
        ]
        panel = panel.loc[panel.index.get_level_values("asset").isin(eligible_assets)]
    else:
        eligible_assets = list(prices.columns)

    rebalance_dates = sorted(universe_map.keys())

    # Refresh cache tag after potential column filtering (post-filter)
    cache_tag = f"{start:%Y%m%d}_{end:%Y%m%d}_{len(prices)}x{len(prices.columns)}_{data_sig}"

    adv_df = (
        adv_series.unstack("asset")
        .reindex(prices.index)
        .reindex(columns=prices.columns, fill_value=np.nan)
    )
    adv_notional = adv_df * prices

    windows_cfg = cfg.get("windows", {})
    atr_len = int(windows_cfg.get("atr_len", 14))
    vol_window = int(windows_cfg.get("vol_window", 60))

    atr_cache = cache_dir / f"atr_{atr_len}_{cache_tag}.parquet"
    if atr_cache.exists():
        try:
            atr_df = pd.read_parquet(atr_cache)
        except (OSError, ValueError):
            atr_df = atr(prices, n=atr_len)
    else:
        atr_df = atr(prices, n=atr_len)
        try:
            atr_df.to_parquet(atr_cache, compression="snappy")
        except (OSError, ValueError, ImportError):
            pass

    covariance_cfg = cfg.get("covariance", {}) or {}
    cov_dir = _cov_cache_dir(paths_cfg)
    cov_signature_payload = json.dumps(covariance_cfg, sort_keys=True)
    cov_hash = _stable_hash(cov_signature_payload)
    target_hash = _stable_hash(
        "|".join(dt.isoformat() for dt in rebalance_dates)
    )
    cov_cache_path = cov_dir / f"cov_{vol_window}_{cov_hash}_{target_hash}_{cache_tag}.pkl"
    cov_dict = None
    if cov_cache_path.exists():
        try:
            cov_dict = load(cov_cache_path)
        except (OSError, ValueError, EOFError):
            cov_dict = None
    if cov_dict is None:
        cov_dict = rolling_cov(
            returns,
            window=vol_window,
            method_cfg=covariance_cfg,
            target_dates=rebalance_dates,
        )
        try:
            dump(cov_dict, cov_cache_path)
        except (OSError, ValueError):
            pass

    portfolio_cfg = (cfg.get("portfolio", {}) or {})
    portfolio_method = str(portfolio_cfg.get("method", "hrp")).lower()
    hrp_only_mode = portfolio_method in {"hrp_only", "hrp-only"}
    tda_only_mode = portfolio_method in {"tda_only", "tda-only"}
    meta_blend_meta: Dict[str, object] = {"enabled": False}
    meta_blend_cfg: Optional[Dict[str, object]] = None
    neutral_regime = None

    tda_artifacts_dir = Path(paths_cfg.get("artifacts", "./artifacts")) / "tda"
    tda_artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(paths_cfg.get("reports", "./reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    mapper_cfg: Dict[str, object] = {}
    legacy_mapper_cfg = cfg.get("tda_mapper")
    if isinstance(legacy_mapper_cfg, dict):
        mapper_cfg.update(legacy_mapper_cfg)
    new_mapper_cfg = cfg.get("mapper")
    if isinstance(new_mapper_cfg, dict):
        mapper_cfg.update(new_mapper_cfg)

    factors_cfg = cfg.get("factors", {}) or {}

    mapper_params = {
        "n_cubes": int(mapper_cfg.get("n_cubes", 8)),
        "overlap": float(mapper_cfg.get("overlap", 0.4)),
        "lens": str(mapper_cfg.get("lens", "pca_umap")),
        "min_cluster_size": int(mapper_cfg.get("min_cluster_size", 3)),
        "eps_quantile": float(mapper_cfg.get("eps_quantile", 0.25)),
        "epsilon_adaptive": bool(mapper_cfg.get("epsilon_adaptive", True)),
        "random_state": mapper_cfg.get("random_state"),
    }
    if mapper_params["random_state"] is None:
        mapper_params["random_state"] = seed
    mapper_lookback = int(mapper_cfg.get("lookback", max(vol_window, 126)))
    mapper_min_history = int(
        mapper_cfg.get(
            "min_history",
            max(30, mapper_params["min_cluster_size"] * 3),
        )
    )
    periphery_cfg = mapper_cfg.get("peripherality", {}) or {}
    use_peripherality = bool(
        factors_cfg.get(
            "use_peripherality",
            periphery_cfg.get("enabled", True),
        )
    )
    periphery_delta = float(
        factors_cfg.get(
            "delta",
            periphery_cfg.get("delta", 0.15),
        )
    )

    peripherality_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    mapper_metrics_records: List[Dict[str, object]] = []
    mapper_metrics_df: Optional[pd.DataFrame] = None
    mapper_results_by_date: Dict[pd.Timestamp, Dict[str, object]] = {}

    alpha = beta = gamma = 0.0
    regime_gain_effective = 1.0
    regime_mode_effective = "tanh"

    regime_series = pd.Series(0.0, index=prices.index, dtype=float)
    regime_z_series = pd.Series(0.0, index=prices.index, dtype=float)
    momentum_raw_full: Optional[pd.DataFrame] = None
    quality_raw_full: Optional[pd.DataFrame] = None
    momentum_df: Optional[pd.DataFrame] = None
    quality_df: Optional[pd.DataFrame] = None
    mix_df: Optional[pd.DataFrame] = None

    tda_meta: Dict[str, object] = {}
    tda_meta["mapper_params"] = {
        **mapper_params,
        "lookback": mapper_lookback,
        "min_history": mapper_min_history,
    }
    tda_meta["peripherality"] = {
        "enabled": use_peripherality,
        "delta": periphery_delta,
    }
    tda_meta["mapper_artifacts"] = []
    tda_meta["mode"] = portfolio_method

    if hrp_only_mode:
        neutral_regime = float(portfolio_cfg.get("hrp_only_regime_value", 0.5))
        neutral_regime = float(np.clip(neutral_regime, 0.0, 1.0))
        logger.info(
            "Portfolio method '%s' active; using pure HRP weights with constant regime %.3f",
            portfolio_method,
            neutral_regime,
        )
        regime_series = pd.Series(neutral_regime, index=prices.index, dtype=float)
        regime_z_series = pd.Series(0.0, index=prices.index, dtype=float)
        mix_df = pd.DataFrame(1.0, index=prices.index, columns=prices.columns, dtype=float)
        peripherality_df.loc[:, :] = 0.0
        tda_meta.update(
            {
                "mode": "hrp_only",
                "regime_stats": {
                    "min": neutral_regime,
                    "max": neutral_regime,
                    "mean": neutral_regime,
                    "std": 0.0,
                },
            }
        )
        momentum_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
        quality_df = momentum_df.copy()
    else:
        alpha, beta, gamma = get_alphas_from_cfg(cfg)
        logger.info("Alphas used: alpha=%.3f beta=%.3f gamma=%.3f", alpha, beta, gamma)

        regime_gain_cfg = factors_cfg.get("regime_gain")
        regime_mode_cfg = factors_cfg.get("regime_mode")
        env_gain = os.getenv("REGIME_GAIN")
        env_mode = os.getenv("REGIME_MODE")
        gain_source = regime_gain_cfg if regime_gain_cfg is not None else env_gain
        try:
            regime_gain_effective = float(gain_source) if gain_source is not None else 1.0
        except (TypeError, ValueError):
            logger.warning("Invalid regime gain override '%s'; defaulting to 1.0", gain_source)
            regime_gain_effective = 1.0
        mode_source = regime_mode_cfg if regime_mode_cfg is not None else (env_mode or "tanh")
        regime_mode_effective = str(mode_source).lower()
        if regime_mode_effective not in {"tanh", "linear", "power"}:
            logger.warning("Invalid regime mode override '%s'; defaulting to 'tanh'", mode_source)
            regime_mode_effective = "tanh"
        tda_meta["factor_weights"] = {"alpha": alpha, "beta": beta, "gamma": gamma}
        tda_meta["regime_modifiers"] = {
            "gain": regime_gain_effective,
            "mode": regime_mode_effective,
        }

        logger.info("Computing PH regime index via turbulence pipeline")
        regime_base = compute_ph_regime_index(returns, cfg)
        regime_attrs = getattr(regime_base, "attrs", {}) or {}

        z_attr = regime_attrs.get("zscore_series")
        if isinstance(z_attr, pd.Series):
            regime_z_series = z_attr.reindex(prices.index).ffill().fillna(0.0)
        else:
            regime_z_series = pd.Series(0.0, index=prices.index, dtype=float)

        regime_series = regime_base.reindex(prices.index).ffill().fillna(0.0)
        regime_stats = {
            "min": float(regime_series.min()),
            "max": float(regime_series.max()),
            "mean": float(regime_series.mean()),
            "std": float(regime_series.std(ddof=0)),
        }
        tda_meta["regime_stats"] = regime_stats

        regime_flags: Dict[str, pd.Series] = {}
        for key in ("is_alert", "is_riskoff"):
            flag = regime_attrs.get(key)
            if isinstance(flag, pd.Series):
                regime_flags[key] = flag.reindex(prices.index).fillna(False)

        regime_series.attrs = {}
        for name, series in regime_flags.items():
            regime_series.attrs[name] = series
        regime_series.attrs["zscore_series"] = regime_z_series

        regime_report = pd.DataFrame({"regime": regime_series, "zscore": regime_z_series})
        for name, flag_series in regime_flags.items():
            regime_report[name] = flag_series.astype(bool)
        regime_csv_path = reports_dir / f"ph_regime_{cache_tag_full}.csv"
        try:
            regime_report.to_csv(regime_csv_path, index=True)
            tda_meta["regime_report"] = str(regime_csv_path)
        except OSError as exc:
            logger.warning("Failed to write regime report '%s': %s", regime_csv_path, exc)

        momentum_cache = cache_dir / f"momentum_full_{cache_tag_full}.parquet"
        quality_cache = cache_dir / f"quality_full_{cache_tag_full}.parquet"

        def _load_or_compute(path: Path, compute_fn):
            if path.exists():
                try:
                    frame = pd.read_parquet(path)
                    try:
                        frame = frame.rename(columns=str)
                    except Exception:
                        pass
                    return frame
                except Exception:
                    pass
            frame = compute_fn()
            try:
                tmp = frame.copy()
                tmp.columns = tmp.columns.astype(str)
                tmp.to_parquet(path, compression="snappy")
            except (OSError, ValueError, ImportError):
                pass
            return frame

        momentum_raw_full = _load_or_compute(momentum_cache, lambda: momentum_12_1(prices_full))
        quality_raw_full = _load_or_compute(quality_cache, lambda: quality_proxy(prices_full))

        momentum_df = momentum_raw_full.reindex(prices.index, method=None).ffill().fillna(0.0)
        quality_df = quality_raw_full.reindex(prices.index, method=None).ffill().fillna(0.0)

        mapper_cache_dir = _mapper_cache_dir(paths_cfg)

        for date in rebalance_dates:
            if date not in prices.index:
                logger.debug(
                    "Skipping mapper build on %s: date not in price index",
                    date.date(),
                )
                continue
            lookback_returns = returns.loc[:date].tail(mapper_lookback)
            lookback_returns = lookback_returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
            if lookback_returns.shape[0] < mapper_min_history or lookback_returns.shape[1] < 2:
                continue
            regime_value = float(np.clip(regime_series.loc[date], 0.0, 1.0))
            cache_path = _mapper_cache_path(
                mapper_cache_dir,
                date,
                lookback_returns,
                mapper_params,
                regime_value,
                mapper_lookback,
                mapper_min_history,
            )
            cached_mapper = None
            if cache_path is not None and cache_path.exists():
                cached_mapper = _load_mapper_cache(cache_path)
            if cached_mapper and isinstance(cached_mapper.get("peripherality"), pd.Series):
                if date not in peripherality_df.index:
                    logger.debug(
                        "Skipping cached mapper reuse on %s: date not present in price index",
                        date.date(),
                    )
                    continue
                cached_series: pd.Series = cached_mapper["peripherality"].reindex(
                    peripherality_df.columns,
                    fill_value=0.0,
                )
                peripherality_df.loc[date, cached_series.index] = cached_series
                payload_cached = cached_mapper.get("payload") or {}
                centrality_cached = cached_mapper.get("centrality") or {}
                mapper_graph = None
                if payload_cached:
                    try:
                        mapper_stub = RegimeAwareMapper(**mapper_params)
                        mapper_graph = mapper_stub._build_graph(payload_cached)
                    except Exception:
                        mapper_graph = None
                mapper_results_by_date[date] = {
                    "graph": mapper_graph,
                    "centrality": centrality_cached,
                    "payload": payload_cached,
                }
                summary_cached = cached_mapper.get("metrics_summary") or {}
                metrics_record = {
                    "date": date,
                    **{key: float(value) for key, value in summary_cached.items()},
                }
                mapper_metrics_records.append(metrics_record)
                artifacts_cached = cached_mapper.get("artifacts")
                if isinstance(artifacts_cached, dict):
                    artifact_entry = {"date": date.isoformat()}
                    json_path_cached = artifacts_cached.get("json")
                    png_path_cached = artifacts_cached.get("png")
                    if json_path_cached:
                        artifact_entry["json"] = json_path_cached
                    if png_path_cached:
                        artifact_entry["png"] = png_path_cached
                    tda_meta["mapper_artifacts"].append(artifact_entry)
                continue
            try:
                mapper = RegimeAwareMapper(**mapper_params)
                mapper.fit(
                    lookback_returns,
                    regime_value=regime_value,
                )
                node_metrics = mapper.metrics_.node_metrics
                node_centrality = {
                    str(node): float(value)
                    for node, value in node_metrics["degree_centrality"].to_dict().items()
                }
                per_asset_centrality = pd.Series(
                    0.0,
                    index=lookback_returns.columns,
                    dtype=float,
                )
                for node, row in node_metrics.iterrows():
                    members = row.get("members", [])
                    if not isinstance(members, list):
                        continue
                    centrality_value = float(row.get("degree_centrality", 0.0) or 0.0)
                    for asset in members:
                        if asset in per_asset_centrality.index:
                            per_asset_centrality.loc[asset] = centrality_value
                periph_series = peripherality_factor(per_asset_centrality)
                peripherality_df.loc[date, periph_series.index] = periph_series.reindex(
                    peripherality_df.columns,
                    fill_value=0.0,
                )

                mapper_results_by_date[date] = {
                    "graph": mapper.graph_,
                    "centrality": node_centrality,
                    "payload": mapper._graph_payload or {},
                }
                metrics_summary_current = {
                    key: float(value) for key, value in mapper.metrics_.summary.items()
                }
                metrics_record = {
                    "date": date,
                    **metrics_summary_current,
                }
                mapper_metrics_records.append(metrics_record)
                artifact_info: Optional[Dict[str, str]] = None
                try:
                    json_path, png_path = mapper.export_graph(tda_artifacts_dir / f"mapper_{date:%Y%m%d}")
                    artifact_info = {
                        "json": str(json_path),
                        "png": str(png_path),
                    }
                    tda_meta["mapper_artifacts"].append(
                        {
                            "date": date.isoformat(),
                            "json": str(json_path),
                            "png": str(png_path),
                        }
                    )
                except Exception as exc:
                    logger.warning("Failed to export mapper graph for %s: %s", date.date(), exc)
                if cache_path is not None:
                    periphery_full = periph_series.reindex(
                        peripherality_df.columns,
                        fill_value=0.0,
                    )
                    _store_mapper_cache(
                        cache_path,
                        periphery_full,
                        node_centrality,
                        mapper._graph_payload or {},
                        metrics_summary_current,
                        artifact_info,
                    )
            except Exception as exc:
                # Mapper pode falhar esporadicamente; manter log em debug para evitar poluicao do output.
                logger.debug("Mapper construction failed on %s: %s", date.date(), exc)

        peripherality_df = peripherality_df.ffill().fillna(0.0)
        if not use_peripherality:
            peripherality_df.loc[:, :] = 0.0

        meta_blend_cfg = factors_cfg.get("meta_blend") if isinstance(factors_cfg, dict) else None
        if meta_blend_cfg and meta_blend_cfg.get("enabled"):
            try:
                horizon = int(meta_blend_cfg.get("horizon", 21))
            except (TypeError, ValueError):
                horizon = 21
            forward_cache = cache_dir / f"forward_{horizon}_{cache_tag_full}.parquet"
            if forward_cache.exists():
                try:
                    fwd_returns_full = pd.read_parquet(forward_cache)
                    try:
                        fwd_returns_full = fwd_returns_full.rename(columns=str)
                    except Exception:
                        pass
                except Exception:
                    fwd_returns_full = forward_returns(prices_full, horizon=horizon)
                    try:
                        _tmp = fwd_returns_full.copy()
                        _tmp.columns = _tmp.columns.astype(str)
                        _tmp.to_parquet(forward_cache, compression="snappy")
                    except (OSError, ValueError, ImportError):
                        pass
            else:
                fwd_returns_full = forward_returns(prices_full, horizon=horizon)
                try:
                    _tmp = fwd_returns_full.copy()
                    _tmp.columns = _tmp.columns.astype(str)
                    _tmp.to_parquet(forward_cache, compression="snappy")
                except (OSError, ValueError, ImportError):
                    pass
            fwd_returns_full = fwd_returns_full.reindex(prices.index)
            asset_list = eligible_assets if eligible_assets else list(prices.columns)
            dataset_cache_id = (
                f"dataset_h{horizon}_reg{int(bool(meta_blend_cfg.get('use_regime_feature', True)))}_"
                f"{meta_blend_cfg.get('model_type', 'ridge')}_{cache_tag}"
            )
            scores_cache_dir = cache_dir / "meta_blend_scores"
            try:
                cfg_payload = json.dumps(meta_blend_cfg, sort_keys=True, default=str)
            except TypeError:
                cfg_payload = str(sorted(meta_blend_cfg.items()))
            scores_cache_id = f"{dataset_cache_id}_{_stable_hash(cfg_payload)}"
            mix_df, meta_blend_meta = run_meta_blend(
                momentum_raw_full.reindex(prices.index).reindex(columns=asset_list),
                quality_raw_full.reindex(prices.index).reindex(columns=asset_list),
                regime_series.reindex(prices.index),
                fwd_returns_full.reindex(columns=asset_list),
                assets=asset_list,
                config_dict=meta_blend_cfg,
                cache_dir=meta_ds_cache_dir,
                cache_id=dataset_cache_id,
                scores_cache_dir=scores_cache_dir,
                scores_cache_id=scores_cache_id,
            )
            mix_df = mix_df.reindex(prices.index).ffill()
            mix_df = mix_df.infer_objects(copy=False).fillna(0.0)
            mix_df = mix_df.reindex(columns=prices.columns, fill_value=0.0)
            if use_peripherality:
                periph_aligned = peripherality_df.reindex(mix_df.index).reindex(columns=mix_df.columns).fillna(0.0)
                mix_df = mix_df + periphery_delta * periph_aligned
            meta_blend_meta.setdefault("enabled", True)
        else:
            common_index = (
                regime_series.index.intersection(momentum_df.index).intersection(quality_df.index)
            )
            if common_index.empty:
                mix_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            else:
                regime_sub = regime_series.reindex(common_index)
                momentum_sub = momentum_df.reindex(common_index)
                quality_sub = quality_df.reindex(common_index)
                periphery_sub = peripherality_df.reindex(common_index)
                mix_df = mix_scores(
                    regime_sub,
                    momentum_sub,
                    quality_sub,
                    alpha,
                    beta,
                    gamma,
                    peripherality=periphery_sub if use_peripherality else None,
                    use_peripherality=use_peripherality,
                    delta=periphery_delta,
                    regime_gain=regime_gain_effective,
                    regime_mode=regime_mode_effective,
                )
                mix_df = mix_df.rename(columns=lambda c: c.replace("mix_", ""))
                mix_df = mix_df.reindex(prices.index).ffill().fillna(0.0)
            meta_blend_meta = {"enabled": False}

        if mapper_metrics_records:
            mapper_metrics_df = pd.DataFrame(mapper_metrics_records)
            mapper_metrics_df["date"] = pd.to_datetime(
                mapper_metrics_df["date"].apply(
                    lambda dt: dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
                ),
                errors="coerce",
            )
            metrics_path = reports_dir / f"mapper_metrics_{cache_tag_full}.csv"
            try:
                mapper_metrics_df.to_csv(metrics_path, index=False)
                tda_meta["mapper_metrics_path"] = str(metrics_path)
            except OSError as exc:
                logger.warning("Failed to write mapper metrics report '%s': %s", metrics_path, exc)
        tda_meta["mapper_dates"] = [dt.isoformat() for dt in sorted(mapper_results_by_date.keys())]

    tda_meta.setdefault("mapper_dates", [])

    if mix_df is None:
        mix_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    # Precompute base HRP weights per rebalance date (graph + order), independent of mix/caps
    cov_dates = sorted(cov_dict.keys())
    def _precompute_hrp(date: pd.Timestamp) -> Tuple[pd.Timestamp, Optional[pd.Series]]:
        try:
            universe_assets = universe_map.get(date, [])
            if not universe_assets:
                return date, None
            cov = _latest_covariance(date, cov_dict, cov_dates)
            if cov is None:
                return date, None
            cov_sub = (
                cov.reindex(index=universe_assets, columns=universe_assets)
                .dropna(axis=0, how="any")
                .dropna(axis=1, how="any")
            )
            if cov_sub.shape[0] < 2:
                return date, None
            graph = _build_mst_from_cov(cov_sub)
            mapper_info = mapper_results_by_date.get(date)
            if mapper_info:
                order = topo_seriation_from_graph(
                    cov_sub,
                    graph,
                    mapper_graph=mapper_info.get("graph"),
                    mapper_centrality=mapper_info.get("centrality"),
                )
            else:
                order = topo_seriation_from_graph(cov_sub, graph)
            hrp_w = hrp_weights_from_order(cov_sub, order)
            return date, hrp_w
        except Exception:
            return date, None

    precomputed_hrp: Dict[pd.Timestamp, pd.Series] = {}
    if rebalance_dates:
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_precompute_hrp)(dt) for dt in rebalance_dates
        )
        precomputed_hrp = {dt: w for dt, w in results if w is not None}

    risk_cfg = cfg.get("risk", {}) or {}

    turnover_cap = float(
        risk_cfg.get(
            "turnover_cap",
            cfg.get("turnover_cap", 0.25),
        )
    )
    target_vol = float(
        risk_cfg.get(
            "target_vol",
            cfg.get("vol_target", 0.10),
        )
    )
    fee_bps = float(cfg.get("costs", {}).get("fee_bps", 5.0))
    slip_params = {
        "k": float(cfg.get("costs", {}).get("k", 0.1)),
        "max_bps": float(cfg.get("costs", {}).get("max_bps", 50.0)),
    }
    # --- parametros de risco usados no kill e na reentrada ---
    mdd_lookback = int(risk_cfg.get("mdd_lookback", 90))
    mdd_thres    = float(risk_cfg.get("mdd_thres", -0.20))
    vol_mult     = float(risk_cfg.get("vol_mult", 1.8))
    cooldown_days = int(risk_cfg.get("cooldown_days", 21))
    reentry_hysteresis = float(risk_cfg.get("reentry_hysteresis", 0.05))  # 5pp

    regime_target_vol_cfg = risk_cfg.get("regime_target_vol")
    if not regime_target_vol_cfg:
        scale_low = risk_cfg.get("regime_scale_low")
        scale_high = risk_cfg.get("regime_scale_high")
        if scale_low is not None or scale_high is not None:
            regime_target_vol_cfg = {}
            if scale_low is not None:
                regime_target_vol_cfg["scale_low"] = float(scale_low)
            if scale_high is not None:
                regime_target_vol_cfg["scale_high"] = float(scale_high)
    regime_gross_cfg = risk_cfg.get("regime_gross")
    participation_cap_base = float(
        risk_cfg.get(
            "participation_cap",
            cfg.get("participation_cap", 0.025),
        )
    )
    participation_cap_regime_cfg = risk_cfg.get("participation_cap_regime")
    max_cluster_regime_cfg = risk_cfg.get("max_cluster_regime")
    max_cluster_static = risk_cfg.get("max_cluster")
    turnover_cap_regime_cfg = risk_cfg.get("turnover_cap_regime")

    # Universe already computed above; reuse it here for the trading loop
    rebalance_dates = sorted(universe_map.keys())

    all_assets = list(prices.columns)
    state = _initial_state(prices.index, all_assets)
    kill_triggered = False
    cooldown = 0  # evita UnboundLocalError e controla a reentrada

    # --- Acumuladores de capacidade ---
    cap_days: int = 0
    cap_bind_days: int = 0
    asset_pre_bind_days: int = 0
    cluster_pre_bind_days: int = 0
    turnover_cut_sum: float = 0.0
    cap_adjustment_l1_total: float = 0.0
    binding_events: List[Dict[str, object]] = []
    asset_pre_bind_history: Dict[pd.Timestamp, bool] = {}
    cluster_pre_bind_history: Dict[pd.Timestamp, bool] = {}
    cap_adjust_history: Dict[pd.Timestamp, bool] = {}
    cap_adjustment_l1_history: Dict[pd.Timestamp, float] = {}
    asset_pre_bind_counter: Counter[str] = Counter()
    asset_adjust_counter: Counter[str] = Counter()
    cluster_pre_bind_counter: Counter[str] = Counter()
    cluster_adjust_counter: Counter[str] = Counter()
    # ----------------------------------
    target_vol_history: Dict[pd.Timestamp, float] = {}
    gross_history: Dict[pd.Timestamp, float] = {}
    participation_cap_history: Dict[pd.Timestamp, float] = {}
    max_cluster_history: Dict[pd.Timestamp, float] = {}
    turnover_cap_history: Dict[pd.Timestamp, float] = {}
    telemetry_by_date: Dict[pd.Timestamp, Dict[str, float]] = {}

    cov_dates = sorted(cov_dict.keys())

    logger.info("Starting backtest loop over %d trading days", len(prices.index))
    for date in prices.index:
        returns_today = returns.loc[date]
        weights_vector = state.current_weights.reindex(all_assets, fill_value=0.0)
        portfolio_ret = float((returns_today * weights_vector).sum())
        state.portfolio_returns.loc[date] = portfolio_ret
        state.equity *= 1.0 + portfolio_ret
        state.equity_curve.loc[date] = state.equity

        rolling_vol = state.portfolio_returns.loc[:date].tail(vol_window).std(ddof=0)
        state.vol_series.loc[date] = (
            rolling_vol * np.sqrt(252) if not np.isnan(rolling_vol) else np.nan
        )

        regime_slice = regime_series.loc[:date]
        regime_value = (
            float(np.clip(regime_slice.iloc[-1], 0.0, 1.0))
            if not regime_slice.empty
            else 0.0
        )

        target_vol_eff, _ = _regime_target_vol(
            target_vol,
            regime_value,
            regime_target_vol_cfg if isinstance(regime_target_vol_cfg, dict) else None,
        )
        target_vol_history[date] = target_vol_eff

        gross_target = _regime_gross_target(
            regime_value,
            regime_gross_cfg if isinstance(regime_gross_cfg, dict) else None,
        )
        gross_history[date] = gross_target

        cap_low, cap_high = _resolve_bounds(
            participation_cap_base,
            participation_cap_regime_cfg if isinstance(participation_cap_regime_cfg, dict) else None,
            minimum=1e-6,
        )
        participation_cap_eff = _lerp(cap_low, cap_high, regime_value)
        participation_cap_history[date] = participation_cap_eff

        if isinstance(max_cluster_regime_cfg, dict):
            cluster_low, cluster_high = _resolve_bounds(
                max_cluster_static if max_cluster_static is not None else max(3.0 * participation_cap_eff, participation_cap_eff),
                max_cluster_regime_cfg,
                minimum=participation_cap_eff,
            )
            max_cluster_eff = _lerp(cluster_low, cluster_high, regime_value)
        elif max_cluster_static is not None:
            max_cluster_eff = float(max_cluster_static)
        else:
            max_cluster_eff = max(3.0 * participation_cap_eff, 1e-6)
        max_cluster_history[date] = max_cluster_eff

        turn_low, turn_high = _resolve_bounds(
            turnover_cap,
            turnover_cap_regime_cfg if isinstance(turnover_cap_regime_cfg, dict) else None,
            minimum=1e-6,
        )
        turnover_cap_eff = _lerp(turn_low, turn_high, regime_value)
        turnover_cap_history[date] = turnover_cap_eff

        telemetry_by_date.setdefault(
            date,
            {
                "regime_value": float(regime_value),
                "target_vol_eff": float(target_vol_eff),
                "gross_target": float(gross_target),
                "participation_cap": float(participation_cap_eff),
                "max_cluster": float(max_cluster_eff),
                "turnover_cap": float(turnover_cap_eff),
                "caps_aplicados": 0.0,
                "bind_rate": float(cap_bind_days / cap_days) if cap_days else 0.0,
                "regime_zscore": float(regime_z_series.loc[date]) if date in regime_z_series.index else float("nan"),
            },
        )

        # --- calculo do MDD em janela e regra de reentrada com histerese ---
        trailing = state.equity_curve.loc[:date].tail(mdd_lookback).dropna()
        if len(trailing) >= 2:
            dd = trailing / trailing.cummax() - 1.0
            rolling_mdd = float(dd.min())
        else:
            rolling_mdd = 0.0
        curr_vol = float(state.vol_series.loc[date]) if not np.isnan(state.vol_series.loc[date]) else np.nan

        regime_ok = guards.regime_cooldown_guard(
            regime_series=regime_series,
            min_reset_days=int(risk_cfg.get("regime_reset_days", 5)),
            alert_threshold=float(risk_cfg.get("regime_alert_sigma", 1.0)),
            last_trigger_date=state.last_kill_date,
            current_date=date,
        ) if kill_triggered else True

        ks_result = guards.rolling_mdd_kill_switch(
            equity=state.equity_curve.loc[:date],
            returns=state.portfolio_returns.loc[:date],
            target_vol=target_vol,
            mdd_lookback=mdd_lookback,
            mdd_threshold=mdd_thres,
            vol_multiplier=vol_mult,
        )

        kill_trigger_fresh = False
        if not kill_triggered and ks_result.active:
            #logger.warning("Kill switch triggered on %s (%s)", date.date(), ks_result.reason)
            kill_triggered = True
            cooldown = max(cooldown_days, ks_result.cooldown)
            state.last_kill_date = date
            kill_trigger_fresh = True

        if kill_triggered:
            prev_weights = state.current_weights.reindex(all_assets, fill_value=0.0)
            if kill_trigger_fresh or float(prev_weights.abs().sum()) > 1e-9:
                # Flatten the book as soon as the kill switch fires to stop further losses.
                target_weights = pd.Series(0.0, index=all_assets)
                trades = _execute_portfolio_trade(
                    date=date,
                    prev_weights=prev_weights,
                    target_weights=target_weights,
                    equity=state.equity,
                    prices=prices,
                    adv_notional=adv_notional,
                    fee_bps=fee_bps,
                    slip_params=slip_params,
                )
                state.trades.extend(trades)
                trade_costs = sum(
                    (trade.get("fees", 0.0) or 0.0) + (trade.get("slip", 0.0) or 0.0)
                    for trade in trades
                )
                state.equity -= trade_costs
                state.current_weights = target_weights
            state.weights_history[date] = state.current_weights.reindex(all_assets, fill_value=0.0)
            if ks_result.active or not regime_ok:
                cooldown = max(0, cooldown - 1)
            else:
                logger.info(
                    "Kill switch lifted on %s (reason=%s, cooldown=%d)",
                    date.date(),
                    ks_result.reason,
                    cooldown,
                )
                kill_triggered = False
                cooldown = 0
            continue

        if date not in rebalance_dates:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        universe_assets = universe_map.get(date, [])
        if not universe_assets:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        logger.info("Rebalance on %s with %d assets", date.date(), len(universe_assets))

        periphery_lambda = float(portfolio_cfg.get("periphery_bias_lambda", 0.0))
        centrality_series = None
        if periphery_lambda > 0.0 and not peripherality_df.empty:
            if date in peripherality_df.index:
                periph_row = peripherality_df.loc[date].reindex(all_assets)
                if periph_row.notna().any():
                    centrality_series = (1.0 - periph_row).clip(lower=0.0)

        ks_flag = risk_controls.kill_switch(
            state.equity_curve.loc[:date],
            state.vol_series.loc[:date],
            target_vol,
            mdd_lookback=mdd_lookback,
            mdd_thres=mdd_thres,
            vol_mult=vol_mult,
        )
        if ks_flag:
            #logger.warning("Kill switch triggered on %s", date.date())
            kill_triggered = True
            cooldown = cooldown_days
            target_weights = pd.Series(0.0, index=all_assets)
        else:
            target_weights, cap_info = _compute_target_weights(
                date=date,
                universe=universe_assets,
                mix_df=mix_df,
                atr_df=atr_df,
                returns=returns,
                cov_dict=cov_dict,
                cov_dates=cov_dates,
                prices=prices,
                alpha_weights=state.current_weights,
                cfg=cfg,
                regime_value=regime_value,
                max_asset=participation_cap_eff,
                max_cluster=max_cluster_eff,
                target_vol=target_vol_eff,
                precomputed_hrp=precomputed_hrp,
                centrality=centrality_series,
                periphery_lambda=periphery_lambda,
            )

        if target_weights is None:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        target_weights = target_weights * gross_target

        # --- metricas de binding de participation cap (por-ativo/cluster) ---
        cap_info = cap_info or {}
        cap_days += 1
        cap_adjusted = bool(cap_info.get("cap_bind", False))
        if cap_adjusted:
            cap_bind_days += 1
        cap_adjust_history[date] = cap_adjusted

        asset_hits = cap_info.get("asset_pre_bind_assets") or []
        cluster_hits = cap_info.get("cluster_pre_bind_clusters") or []
        adjusted_assets = cap_info.get("asset_adjusted_assets") or []
        adjusted_clusters = cap_info.get("cluster_bind_clusters") or []

        asset_hits = [str(asset) for asset in asset_hits]
        cluster_hits = [str(cluster) for cluster in cluster_hits]
        adjusted_assets = [str(asset) for asset in adjusted_assets]
        adjusted_clusters = [str(cluster) for cluster in adjusted_clusters]

        if asset_hits:
            asset_pre_bind_days += 1
        if cluster_hits:
            cluster_pre_bind_days += 1

        asset_pre_bind_history[date] = bool(asset_hits)
        cluster_pre_bind_history[date] = bool(cluster_hits)
        asset_pre_bind_counter.update(asset_hits)
        cluster_pre_bind_counter.update(cluster_hits)
        asset_adjust_counter.update(adjusted_assets)
        cluster_adjust_counter.update(adjusted_clusters)

        adjustment_l1 = float(cap_info.get("cap_adjustment_l1", 0.0) or 0.0)
        cap_adjustment_l1_total += adjustment_l1
        cap_adjustment_l1_history[date] = adjustment_l1

        binding_events.append(
            {
                "date": date.isoformat(),
                "cap_bind": cap_adjusted,
                "asset_pre_bind": asset_hits,
                "asset_adjusted": adjusted_assets,
                "cluster_pre_bind": cluster_hits,
                "cluster_adjusted": adjusted_clusters,
                "cap_adjustment_l1": adjustment_l1,
                "expected_tilt_status": cap_info.get("expected_tilt_status"),
                "expected_tilt_linf": float(cap_info.get("expected_tilt_linf", 0.0) or 0.0),
                "max_asset": cap_info.get("max_asset"),
                "max_cluster": cap_info.get("max_cluster"),
            }
        )
        telemetry_by_date[date].update(
            {
                "regime_value": float(regime_value),
                "target_vol_eff": float(target_vol_eff),
                "gross_target": float(gross_target),
                "participation_cap": float(participation_cap_eff),
                "max_cluster": float(max_cluster_eff),
                "turnover_cap": float(turnover_cap_eff),
                "caps_aplicados": float(1.0 if cap_adjusted else 0.0),
                "bind_rate": float(cap_bind_days / cap_days) if cap_days else 0.0,
                "regime_zscore": float(regime_z_series.loc[date]) if date in regime_z_series.index else float("nan"),
            }
        )
        # ---------------------------------------------------------------------
        # --- turnover cap: medir corte fracionario ---
        aligned_prev = state.current_weights.reindex(target_weights.index).fillna(0.0)
        raw_turnover = 0.5 * (target_weights - aligned_prev).abs().sum()
        target_weights = risk_controls.apply_turnover_cap(
            state.current_weights.reindex(target_weights.index, fill_value=0.0),
            target_weights,
            cap=turnover_cap_eff,
        )
        post_turnover = 0.5 * (target_weights - aligned_prev).abs().sum()
        if raw_turnover > 1e-12:
            cut_frac = max(0.0, 1.0 - float(post_turnover / raw_turnover))
            turnover_cut_sum += cut_frac
        # ---------------------------------------------
        target_weights = target_weights.reindex(all_assets, fill_value=0.0)

        trades = _execute_portfolio_trade(
            date=date,
            prev_weights=state.current_weights,
            target_weights=target_weights,
            equity=state.equity,
            prices=prices,
            adv_notional=adv_notional,
            fee_bps=fee_bps,
            slip_params=slip_params,
        )
        state.trades.extend(trades)
        trade_costs = sum(
            (trade.get("fees", 0.0) or 0.0) + (trade.get("slip", 0.0) or 0.0)
            for trade in trades
        )
        state.equity -= trade_costs
        state.current_weights = target_weights
        state.weights_history[date] = state.current_weights.reindex(
            all_assets, fill_value=0.0
        )
        logger.info(
            "Post-trade equity %.4f; net turnover %.2f%%",
            state.equity,
            sum(abs(trade["fill_qty"]) for trade in trades),
        )

    weight_df = (
        pd.DataFrame(state.weights_history).T.reindex(prices.index).ffill().fillna(0.0)
    )
    equity_curve = state.equity_curve
    trades_df = pd.DataFrame(state.trades)

    kpis = _compute_kpis(state.portfolio_returns, equity_curve)

    # --- resumo de capacidade para consumers (capacity.py, etc.) ---
    cap_summary = {
        "cap_days": int(cap_days),
        "cap_bind_days": int(cap_bind_days),
        "cap_bind_rate": (cap_bind_days / cap_days) if cap_days else 0.0,
        "avg_turnover_cut_frac": (turnover_cut_sum / cap_days) if cap_days else 0.0,
        "asset_pre_bind_days": int(asset_pre_bind_days),
        "asset_pre_bind_rate": (asset_pre_bind_days / cap_days) if cap_days else 0.0,
        "cluster_pre_bind_days": int(cluster_pre_bind_days),
        "cluster_pre_bind_rate": (cluster_pre_bind_days / cap_days) if cap_days else 0.0,
        "avg_cap_adjustment_l1": (cap_adjustment_l1_total / cap_days) if cap_days else 0.0,
    }
    # ----------------------------------------------------------------

    def _history_stats(history: Dict[pd.Timestamp, float]) -> Dict[str, float]:
        if not history:
            return {}
        arr = np.asarray(list(history.values()), dtype=float)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    def _ordered_counts(counter: Counter[str]) -> Dict[str, int]:
        if not counter:
            return {}
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return {key: int(value) for key, value in ordered}

    def _frequency(counter: Counter[str], denom: int) -> Dict[str, float]:
        if not counter or denom <= 0:
            return {}
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return {key: value / denom for key, value in ordered}

    binding_events_serialized = [
        {
            "date": event["date"],
            "cap_bind": bool(event["cap_bind"]),
            "asset_pre_bind": list(event["asset_pre_bind"]),
            "asset_adjusted": list(event["asset_adjusted"]),
            "cluster_pre_bind": list(event["cluster_pre_bind"]),
            "cluster_adjusted": list(event["cluster_adjusted"]),
            "cap_adjustment_l1": float(event["cap_adjustment_l1"]),
            "expected_tilt_status": event.get("expected_tilt_status"),
            "expected_tilt_linf": float(event.get("expected_tilt_linf", 0.0)),
            "max_asset": event.get("max_asset"),
            "max_cluster": event.get("max_cluster"),
        }
        for event in binding_events
    ]
    cap_bind_history_serialized = {
        dt.isoformat(): bool(flag) for dt, flag in cap_adjust_history.items()
    }
    cap_adjustment_l1_history_serialized = {
        dt.isoformat(): float(value) for dt, value in cap_adjustment_l1_history.items()
    }

    regime_meta = {
        "target_vol_effective": _history_stats(target_vol_history),
        "gross_exposure": _history_stats(gross_history),
        "participation_cap": _history_stats(participation_cap_history),
        "max_cluster_cap": _history_stats(max_cluster_history),
        "turnover_cap": _history_stats(turnover_cap_history),
    }
    regime_meta["per_date"] = {
        dt.isoformat(): {key: float(value) for key, value in metrics.items()}
        for dt, metrics in telemetry_by_date.items()
    }
    binding_meta = {
        "summary": {
            "cap_bind_rate": cap_summary["cap_bind_rate"],
            "asset_pre_bind_rate": cap_summary["asset_pre_bind_rate"],
            "cluster_pre_bind_rate": cap_summary["cluster_pre_bind_rate"],
            "avg_cap_adjustment_l1": cap_summary["avg_cap_adjustment_l1"],
        },
        "per_date": binding_events_serialized,
        "cap_bind_history": cap_bind_history_serialized,
        "cap_adjustment_l1_history": cap_adjustment_l1_history_serialized,
        "asset_pre_bind_counts": _ordered_counts(asset_pre_bind_counter),
        "asset_adjusted_counts": _ordered_counts(asset_adjust_counter),
        "cluster_pre_bind_counts": _ordered_counts(cluster_pre_bind_counter),
        "cluster_adjusted_counts": _ordered_counts(cluster_adjust_counter),
        "asset_pre_bind_frequency": _frequency(asset_pre_bind_counter, cap_days),
        "asset_adjusted_frequency": _frequency(asset_adjust_counter, cap_days),
        "cluster_pre_bind_frequency": _frequency(cluster_pre_bind_counter, cap_days),
        "cluster_adjusted_frequency": _frequency(cluster_adjust_counter, cap_days),
    }
    regime_meta["binding"] = binding_meta

    regime_controls_path: Optional[Path] = None
    if telemetry_by_date:
        telemetry_df = pd.DataFrame.from_dict(
            {pd.Timestamp(dt): vals for dt, vals in telemetry_by_date.items()},
            orient="index",
        ).sort_index()
        telemetry_df.index.name = "date"
        if "regime_zscore" in telemetry_df.columns:
            telemetry_df = telemetry_df.rename(columns={"regime_zscore": "zscore"})
        else:
            telemetry_df["zscore"] = regime_z_series.reindex(telemetry_df.index).astype(float)
        if "regime_value" in telemetry_df.columns:
            telemetry_df["regime_value"] = telemetry_df["regime_value"].astype(float)
        if "target_vol_eff" in telemetry_df.columns:
            telemetry_df["target_vol_eff"] = telemetry_df["target_vol_eff"].astype(float)

        mapper_summary = None
        if mapper_metrics_df is not None and not mapper_metrics_df.empty:
            mapper_summary = (
                mapper_metrics_df.dropna(subset=["date"])
                .set_index("date")[["n_components", "avg_degree", "gini_node_size"]]
                .sort_index()
            )
        if mapper_summary is not None:
            regime_controls = telemetry_df.join(mapper_summary, how="left")
        else:
            regime_controls = telemetry_df

        for column in ("n_components", "avg_degree", "gini_node_size"):
            if column not in regime_controls.columns:
                regime_controls[column] = np.nan

        if "regime_value" not in regime_controls.columns:
            regime_controls["regime_value"] = regime_series.reindex(regime_controls.index).astype(float)
        if "target_vol_eff" not in regime_controls.columns:
            regime_controls["target_vol_eff"] = np.nan

        regime_controls = regime_controls.sort_index()
        desired_columns = [
            "zscore",
            "regime_value",
            "target_vol_eff",
            "n_components",
            "avg_degree",
            "gini_node_size",
        ]
        for column in desired_columns:
            if column not in regime_controls.columns:
                if column == "zscore":
                    regime_controls[column] = regime_z_series.reindex(regime_controls.index).astype(float)
                elif column == "regime_value":
                    regime_controls[column] = regime_series.reindex(regime_controls.index).astype(float)
                else:
                    regime_controls[column] = np.nan
        regime_controls = regime_controls[desired_columns]

        meta_dir = Path(paths_cfg.get("artifacts", "./artifacts")) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        regime_controls_path = meta_dir / f"regime_controls_{cache_tag_full}.csv"
        try:
            regime_controls.to_csv(regime_controls_path, index=True)
            tda_meta["regime_controls_path"] = str(regime_controls_path)
        except OSError as exc:
            logger.warning("Failed to write regime controls telemetry '%s': %s", regime_controls_path, exc)
    if regime_controls_path is None:
        tda_meta.setdefault("regime_controls_path", None)

    portfolio_meta = {"method": portfolio_method}
    if tda_only_mode:
        portfolio_meta["base_allocation"] = "uniform"
    else:
        portfolio_meta["base_allocation"] = "hrp"
    if hrp_only_mode and neutral_regime is not None:
        portfolio_meta["regime_constant"] = float(neutral_regime)

    meta_blend_artifact = None
    if meta_blend_cfg and meta_blend_cfg.get("save_artifacts") and meta_blend_meta.get("enabled"):
        meta_blend_artifact = _save_meta_blend_meta(meta_blend_meta, cfg)
        if meta_blend_artifact is not None:
            meta_blend_meta["artifact_path"] = str(meta_blend_artifact)
    return {
        "equity_curve": equity_curve,
        "daily_positions": weight_df,
        "trades": trades_df,
        "weights": state.current_weights,
        "kpis": kpis,
        "meta": {
            "start": start,
            "end": end,
            "n_days": len(prices.index),
            "n_assets": len(all_assets),
            "rebalance_dates": len(rebalance_dates),
            "capacity": cap_summary,
            "regime_controls": regime_meta,
            "portfolio": portfolio_meta,
            "meta_blend": meta_blend_meta,
            "tda_params": tda_meta,
            "tda_topology": tda_meta,
            "regime_stats": tda_meta.get("regime_stats", {}),
            "tfi_stats": tda_meta.get("regime_stats", {}),
        },
    }


def _initial_state(index: pd.Index, assets: Iterable[str]) -> BacktestState:
    zero_weights = pd.Series(0.0, index=list(assets), dtype=float)
    return BacktestState(
        equity=1.0,
        current_weights=zero_weights,
        portfolio_returns=pd.Series(index=index, dtype=float),
        equity_curve=pd.Series(index=index, dtype=float),
        vol_series=pd.Series(index=index, dtype=float),
        weights_history={},
        trades=[],
    )


def _compute_target_weights(
    date: pd.Timestamp,
    universe: List[str],
    mix_df: pd.DataFrame,
    atr_df: pd.DataFrame,
    returns: pd.DataFrame,
    cov_dict: Dict[pd.Timestamp, pd.DataFrame],
    cov_dates: List[pd.Timestamp],
    prices: pd.DataFrame,
    alpha_weights: pd.Series,
    cfg: Dict,
    regime_value: float,
    max_asset: float,
    max_cluster: float,
    target_vol: float,
    precomputed_hrp: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    centrality: Optional[pd.Series] = None,
    periphery_lambda: float = 0.0,
) -> Tuple[Optional[pd.Series], Dict[str, float]]:
    cap_info: Dict[str, float] = {}
    max_asset = max(float(max_asset), 1e-6)
    max_cluster = max(float(max_cluster), max_asset)
    risk_cfg = cfg.get("risk", {}) or {}
    clusters = cfg.get("clusters")

    portfolio_cfg_local = (cfg.get("portfolio", {}) or {})
    method_local = str(portfolio_cfg_local.get("method", "hrp")).lower()
    hrp_only_mode = method_local in {"hrp_only", "hrp-only"}
    tda_only_mode = method_local in {"tda_only", "tda-only"}

    cov = _latest_covariance(date, cov_dict, cov_dates)
    if cov is None:
        return None, {}

    cov = (
        cov.reindex(index=universe, columns=universe)
        .dropna(axis=0, how="any")
        .dropna(axis=1, how="any")
    )
    if cov.shape[0] < 2:
        logger.warning("Insufficient covariance coverage on %s", date.date())
        return None, {}

    covariance_cfg_local = cfg.get("covariance")
    if not isinstance(covariance_cfg_local, dict):
        covariance_cfg_local = {}
    reuse_cov_for_scaling = not covariance_cfg_local

    pre_w = None
    if precomputed_hrp is not None:
        pre_w = precomputed_hrp.get(date)
        if isinstance(pre_w, pd.Series):
            # Ensure alignment to current universe
            pre_w = pre_w.reindex(cov.index).dropna()
            if pre_w.sum() != 0:
                pre_w = pre_w / pre_w.sum()
    if pre_w is not None and not pre_w.empty and len(pre_w) >= 1:
        hrp_weights = pre_w
    else:
        graph = _build_mst_from_cov(cov)
        mapper_info = mapper_results_by_date.get(date)
        if mapper_info:
            order = topo_seriation_from_graph(
                cov,
                graph,
                mapper_graph=mapper_info.get("graph"),
                mapper_centrality=mapper_info.get("centrality"),
            )
        else:
            order = topo_seriation_from_graph(cov, graph)
        hrp_weights = hrp_weights_from_order(cov, order)
    if tda_only_mode and len(hrp_weights) > 0:
        hrp_weights = pd.Series(1.0 / len(hrp_weights), index=hrp_weights.index)
    # --- ABLATION: ignorar HRP (peso = 1/N) ---
    if os.getenv("ABLATE_NO_HRP", "0") == "1":
        hrp_weights = pd.Series(1.0 / len(hrp_weights), index=hrp_weights.index)

    if periphery_lambda > 0.0:
        centrality_aligned = None
        if centrality is not None:
            centrality_aligned = centrality.reindex(hrp_weights.index)
        hrp_weights = apply_periphery_bias(hrp_weights, centrality_aligned, periphery_lambda)

    # mix_row = mix_df.loc[:date].tail(1)
    # if mix_row.empty:
    #     mix_adjusted = pd.Series(1.0, index=hrp_weights.index)
    # else:
    #     mix_row = mix_row.iloc[0].reindex(hrp_weights.index).fillna(0.0)
    #     mix_adjusted = (1.0 + mix_row).clip(lower=0.0)
    #     if mix_adjusted.sum() == 0:
    #         mix_adjusted = pd.Series(1.0, index=hrp_weights.index)

    # blended = hrp_weights * mix_adjusted
    # if blended.sum() == 0:
    #     blended = hrp_weights
    # blended /= blended.sum()
    mix_row = mix_df.loc[:date].tail(1)
    signal_series = (
        mix_row.iloc[0].reindex(hrp_weights.index).fillna(0.0)
        if not mix_row.empty
        else pd.Series(0.0, index=hrp_weights.index, dtype=float)
    )
    if float(signal_series.abs().sum()) < 1e-12:
        trailing = returns.loc[:date].tail(63)
        if not trailing.empty:
            momentum_mu = trailing.mean().reindex(hrp_weights.index).fillna(0.0)
            signal_series = momentum_mu
    expected_cfg = portfolio_cfg_local.get("expected_returns") or {}
    expected_enabled = bool(expected_cfg.get("enabled", False)) and not hrp_only_mode
    max_tilt_cfg = float(expected_cfg.get("max_tilt", 0.2)) if expected_enabled else 0.0
    min_signal_cfg = float(expected_cfg.get("min_signal_z", 0.0))
    regime_gate_cfg = float(expected_cfg.get("regime_gate", 0.0))
    risk_aversion_cfg = float(expected_cfg.get("risk_aversion", 1.0))
    long_only_cfg = bool(expected_cfg.get("long_only", True))
    mu_centered = signal_series - signal_series.mean()
    mu_std = float(mu_centered.std(ddof=0))
    if mu_std > 1e-12:
        mu_z = mu_centered / mu_std
    else:
        mu_z = pd.Series(0.0, index=signal_series.index, dtype=float)
    signal_strength = float(mu_z.abs().mean())
    blended = hrp_weights.copy()
    tilt_status = "disabled" if not expected_enabled else "ready"
    if expected_enabled:
        if signal_strength < (min_signal_cfg - 1e-9):
            tilt_status = "weak_signal"
        elif regime_value < (regime_gate_cfg - 1e-9):
            tilt_status = "regime_gate"
        else:
            try:
                cov_sub = cov.loc[hrp_weights.index, hrp_weights.index]
                blended = expected_sharpe_tilt(
                    hrp_weights,
                    mu_z,
                    cov_sub,
                    max_tilt=max_tilt_cfg,
                    risk_aversion=max(risk_aversion_cfg, 1e-6),
                    long_only=long_only_cfg,
                )
                tilt_status = "applied"
            except Exception as err:
                logger.warning(
                    "Expected-return tilt failed on %s: %s", date.date(), err
                )
                blended = hrp_weights.copy()
                tilt_status = "error"
    if tilt_status != "applied":
        if hrp_only_mode:
            mix_adj = pd.Series(1.0, index=hrp_weights.index, dtype=float)
        else:
            if mix_row.empty:
                mix_adj = pd.Series(1.0, index=hrp_weights.index, dtype=float)
            else:
                s = signal_series
                factors_cfg = cfg.get("factors", {}) or {}
                T_env = os.getenv("SOFTMAX_T")
                if T_env is not None:
                    T = float(T_env)
                else:
                    base_T = float(factors_cfg.get("softmax_T", 0.7))
                    adaptive_cfg = factors_cfg.get("softmax_adaptive")
                    if isinstance(adaptive_cfg, dict):
                        t_low = float(adaptive_cfg.get("low", adaptive_cfg.get("min", base_T)))
                        t_high = float(adaptive_cfg.get("high", adaptive_cfg.get("max", base_T)))
                        t_low = max(t_low, 1e-6)
                        t_high = max(t_high, t_low)
                        blend = float(np.clip(regime_value, 0.0, 1.0))
                        T = _lerp(t_low, t_high, blend)
                    else:
                        T = base_T
                mix_adj = softmax_with_temperature(s, T)
        blended = hrp_weights * mix_adj
        blended = blended / blended.sum() if blended.sum() != 0 else hrp_weights
    cap_info["expected_tilt_status"] = tilt_status
    cap_info["expected_tilt_signal_strength"] = signal_strength
    cap_info["expected_tilt_regime_value"] = float(regime_value)
    cap_info["expected_tilt_max_tilt"] = max_tilt_cfg
    cap_info["expected_tilt_linf"] = float(np.max(np.abs(blended.reindex(hrp_weights.index) - hrp_weights)))


    #capped = risk_controls.apply_caps(
    # --- detectar se o participation cap/cluster cap ira "bater" ---
    asset_hits = [
        asset
        for asset, weight in blended.items()
        if abs(float(weight)) > (max_asset + 1e-12)
    ]
    asset_bind_pre = bool(asset_hits)
    cluster_pre_hits: List[str] = []
    if clusters:
        for cluster_name, assets in clusters.items():
            assets = [a for a in assets if a in blended.index]
            if not assets:
                continue
            cluster_weight = float(blended.loc[assets].sum())
            if abs(cluster_weight) > (max_cluster + 1e-12):
                cluster_pre_hits.append(str(cluster_name))
    cluster_bind_pre = bool(cluster_pre_hits)
    pre_caps = blended.copy()
    capped = risk_controls.apply_caps(
        blended,
        max_asset=max_asset,
        max_cluster=max_cluster,
        clusters=clusters,
    )
    diff = capped.reindex(pre_caps.index, fill_value=0.0) - pre_caps
    adjusted_flag = bool(np.any(np.abs(diff) > 1e-9))
    cap_info['cap_adjusted'] = adjusted_flag
    cap_info['max_asset'] = float(max_asset)
    cap_info['max_cluster'] = float(max_cluster)
    cap_info['asset_pre_bind_assets'] = [str(asset) for asset in asset_hits]
    cap_info['cluster_pre_bind_clusters'] = cluster_pre_hits
    adjusted_assets = diff.index[np.abs(diff) > 1e-9].tolist()
    cap_info['asset_adjusted_assets'] = [str(asset) for asset in adjusted_assets]
    cap_info['cap_adjustment_l1'] = float(np.abs(diff).sum())
    cluster_adjusted: List[str] = []
    if clusters:
        for cluster_name, assets in clusters.items():
            assets = [a for a in assets if a in pre_caps.index]
            if not assets:
                continue
            before = float(pre_caps.loc[assets].sum())
            after = float(capped.loc[assets].sum())
            if abs(before - after) > 1e-9:
                cluster_adjusted.append(str(cluster_name))
    cap_info['cluster_bind_clusters'] = cluster_adjusted
    cap_info['asset_pre_bind_count'] = len(asset_hits)
    cap_info['cluster_pre_bind_count'] = len(cluster_pre_hits)
    cap_info['asset_adjusted_count'] = len(adjusted_assets)
    cap_info['cluster_adjusted_count'] = len(cluster_adjusted)
    # atr_slice = atr_df.loc[:date]
    # if atr_slice.empty:
    #     logger.warning("ATR unavailable on %s", date.date())
    #     return None

    # risk_norm = atr_risk_normalize(capped, atr_slice.tail(1))
    # scaled = scale_to_vol(risk_norm, returns.loc[:date], target_vol=target_vol)
    # --- ABLATION: pular normalizacao por ATR ---
    if os.getenv("ABLATE_NO_ATR", "0") == "1":
        base_weights = capped
    else:
        atr_slice = atr_df.loc[:date]
        if atr_slice.empty:
            logger.warning("ATR unavailable on %s", date.date())
            return None, {}
        base_weights = atr_risk_normalize(capped, atr_slice.tail(1))
    if reuse_cov_for_scaling:
        cov_active = (
            cov.reindex(index=base_weights.index, columns=base_weights.index)
            .fillna(0.0)
        )
        active_weights = base_weights.reindex(cov_active.index).fillna(0.0)
        if cov_active.empty or active_weights.abs().sum() <= 0:
            raise ValueError("No overlapping assets between weights and covariance")
        portfolio_var = float(active_weights @ cov_active @ active_weights)
        if portfolio_var <= 0:
            raise ValueError("Portfolio variance is non-positive")
        current_vol = np.sqrt(portfolio_var)
        if current_vol <= 0:
            raise ValueError("Portfolio variance is non-positive")
        scale_factor = target_vol / current_vol
        scaled = base_weights * scale_factor
    else:
        # alvo de vol segue ativo (escala uniforme)
        #scaled = scale_to_vol(base_weights, returns.loc[:date], target_vol=target_vol)
        # target_vol e anual no YAML; scale_to_vol espera diaria (cov diaria)
        scaled = scale_to_vol(base_weights, returns.loc[:date], target_vol=target_vol)
    cap_info["asset_pre_bind"] = asset_bind_pre
    cap_info["cluster_pre_bind"] = cluster_bind_pre
    cap_info["cap_bind"] = adjusted_flag
    # devolvemos tambem o dicionario com o flag de binding
    return scaled, cap_info


def _latest_covariance(
    date: pd.Timestamp,
    cov_dict: Dict[pd.Timestamp, pd.DataFrame],
    cov_dates: List[pd.Timestamp],
) -> Optional[pd.DataFrame]:
    eligible = [dt for dt in cov_dates if dt <= date]
    if not eligible:
        return None
    return cov_dict[eligible[-1]]


def _build_mst_from_cov(cov: pd.DataFrame) -> nx.Graph:
    corr = cov.corr().fillna(0.0)
    distance = np.sqrt(0.5 * (1.0 - corr.clip(-1.0, 1.0)))
    graph = nx.Graph()
    for asset in corr.columns:
        graph.add_node(asset)
    for i, asset_i in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            asset_j = corr.columns[j]
            weight = float(distance.iloc[i, j])
            graph.add_edge(asset_i, asset_j, weight=weight)
    return nx.minimum_spanning_tree(graph, weight="weight")


def _execute_portfolio_trade(
    date: pd.Timestamp,
    prev_weights: pd.Series,
    target_weights: pd.Series,
    equity: float,
    prices: pd.DataFrame,
    adv_notional: pd.DataFrame,
    fee_bps: float,
    slip_params: Dict[str, float],
) -> List[Dict[str, float]]:
    trades: List[Dict[str, float]] = []
    price_row = prices.loc[date]
    adv_row = (
        adv_notional.loc[date]
        if date in adv_notional.index
        else pd.Series(index=price_row.index, data=np.nan)
    )

    for asset in target_weights.index:
        prev_w = float(prev_weights.get(asset, 0.0))
        new_w = float(target_weights.get(asset, 0.0))
        if abs(prev_w - new_w) < 1e-9:
            continue

        price = float(price_row.get(asset, np.nan))
        if np.isnan(price) or price == 0:
            continue
        adv_val = float(adv_row.get(asset, np.nan))
        if np.isnan(adv_val) or adv_val <= 0:
            adv_val = abs(new_w * equity) * 10.0

        prev_qty = prev_w * equity / price
        target_qty = new_w * equity / price

        execution = execute_trade(
            prev_qty, target_qty, price, adv_val, fee_bps, slip_params
        )
        execution.update(
            {
                "asset": asset,
                "date": date,
                "price": price,
            }
        )
        trades.append(execution)

    return trades


def _compute_kpis(
    portfolio_returns: pd.Series, equity_curve: pd.Series
) -> Dict[str, float]:
    returns = portfolio_returns.fillna(0.0)
    equity = equity_curve.ffill().dropna()
    if equity.empty:
        return {"final_equity": 1.0}

    ending_equity = float(equity.iloc[-1])
    total_return = ending_equity - 1.0
    periods = len(returns)
    if periods and ending_equity > 0:
        ann_return = ending_equity ** (252 / periods) - 1.0
    else:
        ann_return = np.nan
    ann_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = (
        ann_return / ann_vol if ann_vol > 0 and not np.isnan(ann_return) else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    avg_tuw = avg_time_under_water(equity)
    max_tuw = max_time_under_water(equity)

    return {
        "final_equity": ending_equity,
        "total_return": float(total_return),
        "annual_return": float(ann_return) if not np.isnan(ann_return) else np.nan,
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": max_dd,
        "avg_time_under_water": float(avg_tuw),
        "max_time_under_water": float(max_tuw),
    }




def _save_meta_blend_meta(meta: Dict[str, object], cfg: Dict) -> Optional[Path]:
    if not meta or not meta.get("enabled"):
        return None
    paths_cfg = cfg.get("paths", {}) or {}
    reports_root = Path(paths_cfg.get("reports", "./reports"))
    reports_root.mkdir(parents=True, exist_ok=True)
    factors_cfg = cfg.get("factors", {}) or {}
    meta_cfg = factors_cfg.get("meta_blend", {}) or {}
    prefix = str(meta_cfg.get("artifact_prefix", "meta_blend"))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target_path = reports_root / f"{prefix}_{timestamp}.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return target_path
