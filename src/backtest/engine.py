from __future__ import annotations

"""Core backtesting engine scaffolding."""

import os
import json
import hashlib
from dataclasses import dataclass
import logging
from collections import Counter
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
    TFIParams,
    mix_scores,
    momentum_12_1,
    quality_proxy,
    tfi_score,
    get_alphas_from_cfg,
    forward_returns,
)
from portfolio import (
    expected_sharpe_tilt,
    hrp_weights_from_order,
    rolling_cov,
    topo_seriation_from_graph,
)
from risk import atr, atr_risk_normalize, scale_to_vol
from risk import risk_controls
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


def _stable_hash(payload: str) -> str:
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

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

    prices = panel["close"].unstack("asset").sort_index()
    prices_full = prices

    cache_dir = _factor_cache_dir(paths_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_ds_cache_dir = cache_dir / "meta_blend"
    meta_ds_cache_dir.mkdir(parents=True, exist_ok=True)
    data_sig = _data_signature(paths_cfg)
    cache_tag_full = f"{start:%Y%m%d}_{end:%Y%m%d}_{len(prices_full)}x{len(prices_full.columns)}_{data_sig}"

    returns = prices.pct_change()
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

    # logger.info("Pre-computing factor scores")
    # tfi_params = cfg.get("tda", {})
    # # epsilon pode ser None (modo adaptativo no tda.py)
    # _eps_raw = tfi_params.get("epsilon", 0.5)
    # if _eps_raw is None or str(_eps_raw).lower() in {"none", "null"}:
    #     _eps = None
    # else:
    #     _eps = float(_eps_raw)
    # tfi_cfg = TFIParams(
    #     delay=int(tfi_params.get("delay", 1)),
    #     dim=int(tfi_params.get("dim", 3)),
    #     n_cubes=int(tfi_params.get("n_cubes", 4)),
    #     overlap=float(tfi_params.get("overlap", 0.5)),
    #     epsilon=_eps,
    #     min_samples=int(tfi_params.get("min_samples", 3)),
    #     window=int(tfi_params.get("window", max(vol_window, 63))),
    # )
    portfolio_cfg = (cfg.get("portfolio", {}) or {})
    portfolio_method = str(portfolio_cfg.get("method", "hrp")).lower()
    hrp_only_mode = portfolio_method in {"hrp_only", "hrp-only"}
    tda_only_mode = portfolio_method in {"tda_only", "tda-only"}
    meta_blend_meta: Dict[str, object] = {"enabled": False}
    meta_blend_cfg: Optional[Dict[str, object]] = None
    neutral_regime = None
    tfi_cfg = TFIParams.from_config(cfg, vol_window_fallback=vol_window)
    if hrp_only_mode:
        neutral_regime = float(portfolio_cfg.get("hrp_only_regime_value", 0.5))
        neutral_regime = float(np.clip(neutral_regime, 0.0, 1.0))
        logger.info(
            "Portfolio method '%s' active; using pure HRP weights with constant regime %.3f",
            portfolio_method,
            neutral_regime,
        )
        regime_series = pd.Series(neutral_regime, index=prices.index, dtype=float)
        mix_df = pd.DataFrame(1.0, index=prices.index, columns=prices.columns, dtype=float)
        tfi_meta = {
            "mode": "hrp_only",
            "regime_constant": neutral_regime,
            "delay": tfi_cfg.delay,
            "dim": tfi_cfg.dim,
            "n_cubes": tfi_cfg.n_cubes,
            "overlap": tfi_cfg.overlap,
            "epsilon": tfi_cfg.epsilon,
            "min_samples": tfi_cfg.min_samples,
            "window": tfi_cfg.window,
        }
        tfi_stats = {
            "min": neutral_regime,
            "max": neutral_regime,
            "std": 0.0,
            "mean": neutral_regime,
        }
    else:
        logger.info("Pre-computing factor scores")
        logger.info(
            "TFI used: delay=%s dim=%s n_cubes=%s overlap=%s eps=%s window=%s",
            tfi_cfg.delay,
            tfi_cfg.dim,
            tfi_cfg.n_cubes,
            tfi_cfg.overlap,
            tfi_cfg.epsilon,
            tfi_cfg.window,
        )
        alpha, beta, gamma = get_alphas_from_cfg(cfg)
        logger.info("Alphas used: alpha=%.3f beta=%.3f gamma=%.3f", alpha, beta, gamma)
        factors_cfg = cfg.get("factors", {}) or {}
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
        logger.info(
            "Regime modifiers: gain=%.3f mode=%s (cfg=%s env=%s)",
            regime_gain_effective,
            regime_mode_effective,
            regime_gain_cfg,
            env_mode,
        )
        tfi_meta = {
            "delay": tfi_cfg.delay,
            "dim": tfi_cfg.dim,
            "n_cubes": tfi_cfg.n_cubes,
            "overlap": tfi_cfg.overlap,
            "epsilon": tfi_cfg.epsilon,
            "min_samples": tfi_cfg.min_samples,
            "window": tfi_cfg.window,
        }
        if tda_only_mode:
            tfi_meta["mode"] = "tda_only"
        tfi_signature = (
            f"d{tfi_cfg.delay}_m{tfi_cfg.dim}_c{tfi_cfg.n_cubes}_o{tfi_cfg.overlap}_"
            f"e{tfi_cfg.epsilon}_w{tfi_cfg.window}"
        )
        tfi_cache = cache_dir / f"tfi_{tfi_signature}_{cache_tag_full}.parquet"
        if tfi_cache.exists():
            try:
                tfi_loaded = pd.read_parquet(tfi_cache)
                regime_series = tfi_loaded.iloc[:, 0] if not tfi_loaded.empty else pd.Series(dtype=float)
            except (OSError, ValueError):
                regime_series = tfi_score(prices_full, params=tfi_cfg)
        else:
            regime_series = tfi_score(prices_full, params=tfi_cfg)
            try:
                regime_series.to_frame(name="regime").to_parquet(tfi_cache, compression="snappy")
            except (OSError, ValueError, ImportError):
                pass
        regime_series = regime_series.reindex(prices.index).ffill().fillna(0.0)
        if regime_series.empty:
            tfi_stats = {"min": np.nan, "max": np.nan, "std": np.nan, "mean": np.nan}
            logger.warning(
                "TFI regime series is empty after window=%d; mix_scores will receive zeros.",
                tfi_cfg.window,
            )
        else:
            vals = regime_series.values.astype(float)
            tfi_stats = {
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "std": float(np.nanstd(vals)),
                "mean": float(np.nanmean(vals)),
            }
            logger.info(
                "TFI regime stats: min=%.3f max=%.3f mean=%.3f std=%.3f",
                tfi_stats["min"],
                tfi_stats["max"],
                tfi_stats["mean"],
                tfi_stats["std"],
            )
        momentum_cache = cache_dir / f"momentum_full_{cache_tag_full}.parquet"
        quality_cache = cache_dir / f"quality_full_{cache_tag_full}.parquet"
        if momentum_cache.exists():
            try:
                momentum_raw_full = pd.read_parquet(momentum_cache)
                try:
                    momentum_raw_full = momentum_raw_full.rename(columns=str)
                except Exception:
                    pass
            except Exception:
                momentum_raw_full = momentum_12_1(prices_full)
                try:
                    _tmp = momentum_raw_full.copy()
                    _tmp.columns = _tmp.columns.astype(str)
                    _tmp.to_parquet(momentum_cache, compression="snappy")
                except (OSError, ValueError, ImportError):
                    pass
        else:
            momentum_raw_full = momentum_12_1(prices_full)
            try:
                _tmp = momentum_raw_full.copy()
                _tmp.columns = _tmp.columns.astype(str)
                _tmp.to_parquet(momentum_cache, compression="snappy")
            except (OSError, ValueError, ImportError):
                pass
        if quality_cache.exists():
            try:
                quality_raw_full = pd.read_parquet(quality_cache)
                try:
                    quality_raw_full = quality_raw_full.rename(columns=str)
                except Exception:
                    pass
            except Exception:
                quality_raw_full = quality_proxy(prices_full)
                try:
                    _tmp = quality_raw_full.copy()
                    _tmp.columns = _tmp.columns.astype(str)
                    _tmp.to_parquet(quality_cache, compression="snappy")
                except (OSError, ValueError, ImportError):
                    pass
        else:
            quality_raw_full = quality_proxy(prices_full)
            try:
                _tmp = quality_raw_full.copy()
                _tmp.columns = _tmp.columns.astype(str)
                _tmp.to_parquet(quality_cache, compression="snappy")
            except (OSError, ValueError, ImportError):
                pass
        momentum_df = momentum_raw_full.reindex(prices.index, method=None).ffill().fillna(0.0)
        quality_df = quality_raw_full.reindex(prices.index, method=None).ffill().fillna(0.0)
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
            mix_df, meta_blend_meta = run_meta_blend(
                momentum_raw_full.reindex(prices.index).reindex(columns=asset_list),
                quality_raw_full.reindex(prices.index).reindex(columns=asset_list),
                regime_series.reindex(prices.index),
                fwd_returns_full.reindex(columns=asset_list),
                assets=asset_list,
                config_dict=meta_blend_cfg,
                cache_dir=meta_ds_cache_dir,
                cache_id=dataset_cache_id,
            )
            mix_df = mix_df.reindex(prices.index).ffill().fillna(0.0)
            mix_df = mix_df.reindex(columns=prices.columns, fill_value=0.0)
            meta_blend_meta.setdefault("enabled", True)
        else:
            common_index = (
                regime_series.index.intersection(momentum_df.index).intersection(quality_df.index)
            )
            if common_index.empty:
                mix_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            else:
                regime_series = regime_series.reindex(common_index)
                momentum_df = momentum_df.reindex(common_index)
                quality_df = quality_df.reindex(common_index)
                mix_df = mix_scores(
                    regime_series,
                    momentum_df,
                    quality_df,
                    alpha,
                    beta,
                    gamma,
                    regime_gain=regime_gain_effective,
                    regime_mode=regime_mode_effective,
                )
                mix_df = mix_df.rename(columns=lambda c: c.replace("mix_", ""))
                mix_df = mix_df.reindex(prices.index).ffill().fillna(0.0)
            meta_blend_meta = {"enabled": False}

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

    turnover_cap = float(cfg.get("turnover_cap", 0.25))
    target_vol = float(cfg.get("vol_target", 0.10))
    fee_bps = float(cfg.get("costs", {}).get("fee_bps", 5.0))
    slip_params = {
        "k": float(cfg.get("costs", {}).get("k", 0.1)),
        "max_bps": float(cfg.get("costs", {}).get("max_bps", 50.0)),
    }
    # --- parâmetros de risco usados no kill e na reentrada ---
    risk_cfg = cfg.get("risk", {})
    mdd_lookback = int(risk_cfg.get("mdd_lookback", 90))
    mdd_thres    = float(risk_cfg.get("mdd_thres", -0.20))
    vol_mult     = float(risk_cfg.get("vol_mult", 1.8))
    cooldown_days = int(risk_cfg.get("cooldown_days", 21))
    reentry_hysteresis = float(risk_cfg.get("reentry_hysteresis", 0.05))  # 5pp

    regime_target_vol_cfg = risk_cfg.get("regime_target_vol")
    regime_gross_cfg = risk_cfg.get("regime_gross")
    participation_cap_base = float(cfg.get("participation_cap", 0.025))
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
        if regime_slice.empty:
            regime_value = 0.0
        else:
            regime_value = float(np.clip(regime_slice.iloc[-1], 0.0, 1.0))

        target_vol_eff = target_vol
        if isinstance(regime_target_vol_cfg, dict):
            scale_low = float(regime_target_vol_cfg.get("scale_low", regime_target_vol_cfg.get("low", 0.6)))
            scale_high = float(regime_target_vol_cfg.get("scale_high", regime_target_vol_cfg.get("high", 1.4)))
            scale_low = max(scale_low, 0.0)
            scale_high = max(scale_high, scale_low)
            scale = scale_low + (scale_high - scale_low) * regime_value
            target_vol_eff = max(1e-6, target_vol * max(scale, 0.0))
        target_vol_history[date] = target_vol_eff

        gross_target = 1.0
        if isinstance(regime_gross_cfg, dict):
            gross_low = float(regime_gross_cfg.get("low", regime_gross_cfg.get("min", 0.5)))
            gross_high = float(regime_gross_cfg.get("high", regime_gross_cfg.get("max", 1.0)))
            gross_low = max(gross_low, 0.0)
            gross_high = max(gross_high, gross_low)
            gross_target = float(np.clip(gross_low + (gross_high - gross_low) * regime_value, 0.0, 1.0))
        gross_history[date] = gross_target

        participation_cap_eff = participation_cap_base
        if isinstance(participation_cap_regime_cfg, dict):
            cap_low = float(participation_cap_regime_cfg.get("low", participation_cap_regime_cfg.get("min", participation_cap_base)))
            cap_high = float(participation_cap_regime_cfg.get("high", participation_cap_regime_cfg.get("max", participation_cap_base)))
            cap_low = max(cap_low, 1e-6)
            cap_high = max(cap_high, cap_low)
            participation_cap_eff = cap_low + (cap_high - cap_low) * regime_value
        participation_cap_history[date] = participation_cap_eff

        if isinstance(max_cluster_regime_cfg, dict):
            cluster_low = float(max_cluster_regime_cfg.get("low", max(3.0 * participation_cap_eff, participation_cap_eff)))
            cluster_high = float(max_cluster_regime_cfg.get("high", max(3.0 * participation_cap_eff, participation_cap_eff)))
            cluster_low = max(cluster_low, participation_cap_eff)
            cluster_high = max(cluster_high, cluster_low)
            max_cluster_eff = cluster_low + (cluster_high - cluster_low) * regime_value
        elif max_cluster_static is not None:
            max_cluster_eff = float(max_cluster_static)
        else:
            max_cluster_eff = max(3.0 * participation_cap_eff, 1e-6)
        max_cluster_history[date] = max_cluster_eff

        turnover_cap_eff = turnover_cap
        if isinstance(turnover_cap_regime_cfg, dict):
            turn_low = float(turnover_cap_regime_cfg.get("low", turnover_cap_regime_cfg.get("min", turnover_cap)))
            turn_high = float(turnover_cap_regime_cfg.get("high", turnover_cap_regime_cfg.get("max", turnover_cap)))
            turn_low = max(turn_low, 1e-6)
            turn_high = max(turn_high, turn_low)
            turnover_cap_eff = turn_low + (turn_high - turn_low) * regime_value
        turnover_cap_history[date] = turnover_cap_eff

        # --- cálculo do MDD em janela e regra de reentrada com histerese ---
        trailing = state.equity_curve.loc[:date].tail(mdd_lookback).dropna()
        if len(trailing) >= 2:
            dd = trailing / trailing.cummax() - 1.0
            rolling_mdd = float(dd.min())
        else:
            rolling_mdd = 0.0
        curr_vol = float(state.vol_series.loc[date]) if not np.isnan(state.vol_series.loc[date]) else np.nan

        if kill_triggered:
            cooldown = max(0, cooldown - 1)
            ok_mdd = (rolling_mdd > (mdd_thres + reentry_hysteresis))
            ok_vol = (np.isnan(curr_vol)) or (curr_vol <= vol_mult * target_vol)
            if cooldown == 0 and ok_mdd and ok_vol:
                logger.info("Kill switch lifted on %s (mdd=%.2f, vol_ok=%s)",
                            date.date(), rolling_mdd, str(ok_vol))
                kill_triggered = False

        # if kill_triggered:
        #     state.weights_history[date] = state.current_weights.reindex(
        #         all_assets, fill_value=0.0
        #     )
        #     continue

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

        ks_flag = risk_controls.kill_switch(
            state.equity_curve.loc[:date],
            state.vol_series.loc[:date],
            target_vol,
            mdd_lookback=mdd_lookback,
            mdd_thres=mdd_thres,
            vol_mult=vol_mult,
        )
        if ks_flag:
            logger.warning("Kill switch triggered on %s", date.date())
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
            )

        if target_weights is None:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        target_weights = target_weights * gross_target

        # --- métricas de binding de participation cap (por-ativo/cluster) ---
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
        # ---------------------------------------------------------------------
        # --- turnover cap: medir corte fracionário ---
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
            "tda_params": tfi_meta,
            "tfi_stats": tfi_stats,
            "capacity": cap_summary,
            "regime_controls": regime_meta,
            "portfolio": portfolio_meta,
            "meta_blend": meta_blend_meta,
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
        logger.warning(
            "Skipping rebalance on %s due to missing covariance", date.date()
        )
        return None, {}

    cov = (
        cov.reindex(index=universe, columns=universe)
        .dropna(axis=0, how="any")
        .dropna(axis=1, how="any")
    )
    if cov.shape[0] < 2:
        logger.warning("Insufficient covariance coverage on %s", date.date())
        return None, {}

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
        order = topo_seriation_from_graph(cov, graph)
        hrp_weights = hrp_weights_from_order(cov, order)
    if tda_only_mode and len(hrp_weights) > 0:
        hrp_weights = pd.Series(1.0 / len(hrp_weights), index=hrp_weights.index)
    # --- ABLATION: ignorar HRP (peso = 1/N) ---
    if os.getenv("ABLATE_NO_HRP", "0") == "1":
        hrp_weights = pd.Series(1.0 / len(hrp_weights), index=hrp_weights.index)

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
                        T = t_high - (t_high - t_low) * blend
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
    # --- detectar se o participation cap/cluster cap irá "bater" ---
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
    # --- ABLATION: pular normalização por ATR ---
    if os.getenv("ABLATE_NO_ATR", "0") == "1":
        base_weights = capped
    else:
        atr_slice = atr_df.loc[:date]
        if atr_slice.empty:
            logger.warning("ATR unavailable on %s", date.date())
            return None, {}
        base_weights = atr_risk_normalize(capped, atr_slice.tail(1))
    # alvo de vol segue ativo (escala uniforme)
    #scaled = scale_to_vol(base_weights, returns.loc[:date], target_vol=target_vol)
    # target_vol é anual no YAML; scale_to_vol espera diária (cov diária)
    scaled = scale_to_vol(base_weights, returns.loc[:date], target_vol=target_vol  )##/ np.sqrt(252.0))
    cap_info["asset_pre_bind"] = asset_bind_pre
    cap_info["cluster_pre_bind"] = cluster_bind_pre
    cap_info["cap_bind"] = adjusted_flag
    # devolvemos também o dicionário com o flag de binding
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
