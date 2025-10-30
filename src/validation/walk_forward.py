from __future__ import annotations

"""Walk-forward validation utilities."""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from metrics import avg_time_under_water, cagr, calmar, hit_rate, max_time_under_water, mdd, sharpe, sortino, turnover, vol

__all__ = ["run_walk_forward"]

_IN_SAMPLE_LEN = 252 * 2
_OUT_OF_SAMPLE_LEN = 126


def run_walk_forward(
    cfg: Dict, panel: pd.DataFrame, max_windows: int | None = None
) -> Dict[str, object]:
    """Run a walk-forward evaluation using the rolling-training engine for continuity."""

    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError(
            "panel must be a MultiIndex DataFrame indexed by (date, asset)"
        )

    cfg_run = deepcopy(cfg)
    bt_cfg = cfg_run.setdefault("backtest", {})
    rolling_cfg = deepcopy(bt_cfg.get("rolling_training") or {})
    if not rolling_cfg:
        rolling_cfg = {
            "in_sample_days": _IN_SAMPLE_LEN,
            "out_of_sample_days": _OUT_OF_SAMPLE_LEN,
        }
    rolling_cfg["enabled"] = True
    rolling_cfg.setdefault("in_sample_days", _IN_SAMPLE_LEN)
    rolling_cfg.setdefault("out_of_sample_days", _OUT_OF_SAMPLE_LEN)
    bt_cfg["rolling_training"] = rolling_cfg

    result = run_backtest(cfg_run, panel=panel)

    equity_curve = result.get("equity_curve")
    if equity_curve is None or equity_curve.empty:
        raise ValueError("rolling backtest did not return an equity curve")

    returns = equity_curve.pct_change(fill_method=None).dropna()
    daily_positions = result.get("daily_positions")
    meta = result.get("meta", {}) or {}
    rolling_meta = meta.get("rolling_training") or {}
    windows_meta: List[Dict[str, object]] = rolling_meta.get("windows", []) or []
    if max_windows is not None:
        windows_meta = windows_meta[:max_windows]

    returns_segments: List[pd.Series] = []
    combined_weights: List[pd.DataFrame] = []
    windows_info: List[Dict[str, object]] = []

    for window in windows_meta:
        is_start = pd.Timestamp(window["is_start"])
        is_end = pd.Timestamp(window["is_end"])
        oos_start = pd.Timestamp(window["oos_start"])
        oos_end = pd.Timestamp(window["oos_end"])

        window_returns = returns.loc[oos_start:oos_end]
        if not window_returns.empty:
            returns_segments.append(window_returns)

        if isinstance(daily_positions, pd.DataFrame) and not daily_positions.empty:
            window_weights = daily_positions.loc[oos_start:oos_end]
            if not window_weights.empty:
                combined_weights.append(window_weights)
                turnover_series = turnover(window_weights).dropna()
                window_turnover = (
                    float(turnover_series.mean())
                    if not turnover_series.empty
                    else np.nan
                )
            else:
                window_turnover = np.nan
        else:
            window_weights = None
            window_turnover = np.nan

        window_equity = (1.0 + window_returns).cumprod()
        window_kpis = {
            "CAGR": cagr(window_equity),
            "Sharpe": sharpe(window_returns),
            "Sortino": sortino(window_returns),
            "Vol": vol(window_returns),
            "MaxDD": mdd(window_equity),
            "AvgTimeUnderWater": avg_time_under_water(window_equity),
            "MaxTimeUnderWater": max_time_under_water(window_equity),
            "Calmar": calmar(window_equity),
            "HitRate": hit_rate(window_returns),
            "Turnover": (
                float(window_turnover) if not np.isnan(window_turnover) else np.nan
            ),
        }

        windows_info.append(
            {
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "kpis": window_kpis,
                "meta": window.get("meta", {}),
            }
        )

    combined_returns = (
        pd.concat(returns_segments, copy=False).sort_index()
        if returns_segments
        else pd.Series(dtype=float)
    )
    combined_equity = (1.0 + combined_returns).cumprod()

    if combined_weights:
        weights_df = pd.concat(combined_weights).fillna(0.0)
        turnover_series = turnover(weights_df).dropna()
        turnover_mean = float(turnover_series.mean()) if not turnover_series.empty else np.nan
    else:
        turnover_mean = np.nan

    kpis = {
        "CAGR": cagr(combined_equity),
        "Sharpe": sharpe(combined_returns),
        "Sortino": sortino(combined_returns),
        "Vol": vol(combined_returns),
        "MaxDD": mdd(combined_equity),
        "AvgTimeUnderWater": avg_time_under_water(combined_equity),
        "MaxTimeUnderWater": max_time_under_water(combined_equity),
        "Calmar": calmar(combined_equity),
        "HitRate": hit_rate(combined_returns),
        "Turnover": turnover_mean,
    }

    return {
        "equity_curve": combined_equity,
        "returns": combined_returns,
        "kpis": kpis,
        "windows": windows_info,
    }
