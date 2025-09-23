from __future__ import annotations

"""Walk-forward validation utilities."""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, turnover, vol

__all__ = ["run_walk_forward"]

_IN_SAMPLE_LEN = 252 * 2
_OUT_OF_SAMPLE_LEN = 126


def run_walk_forward(cfg: Dict, panel: pd.DataFrame) -> Dict[str, object]:
    """Run a walk-forward validation with 2y in-sample / 6m out-of-sample windows."""

    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must be a MultiIndex DataFrame indexed by (date, asset)")

    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    all_dates = panel.index.get_level_values(date_level).unique().sort_values()
    if len(all_dates) < (_IN_SAMPLE_LEN + _OUT_OF_SAMPLE_LEN):
        raise ValueError("Panel does not contain enough data for walk-forward validation")

    current_idx = 0
    combined_returns = pd.Series(dtype=float)
    combined_weights: List[pd.DataFrame] = []
    windows_info: List[Dict[str, object]] = []

    while current_idx + _IN_SAMPLE_LEN + _OUT_OF_SAMPLE_LEN <= len(all_dates):
        is_start = all_dates[current_idx]
        is_end = all_dates[current_idx + _IN_SAMPLE_LEN - 1]
        oos_start = all_dates[current_idx + _IN_SAMPLE_LEN]
        oos_end = all_dates[current_idx + _IN_SAMPLE_LEN + _OUT_OF_SAMPLE_LEN - 1]

        panel_slice = panel.loc[(slice(is_start, oos_end), slice(None)), :]

        cfg_window = deepcopy(cfg)
        cfg_window.setdefault("dates", {})
        cfg_window["dates"]["start"] = is_start.isoformat()
        cfg_window["dates"]["end"] = oos_end.isoformat()

        result = run_backtest(cfg_window, panel=panel_slice)
        equity = result["equity_curve"].loc[is_start:oos_end]
        returns = equity.pct_change().fillna(0.0)
        oos_returns = returns.loc[oos_start:oos_end]
        combined_returns = pd.concat([combined_returns, oos_returns])

        weights_oos = result.get("daily_positions")
        if isinstance(weights_oos, pd.DataFrame):
            combined_weights.append(weights_oos.loc[oos_start:oos_end])

        windows_info.append(
            {
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "kpis": result["kpis"],
                "config": {
                    "alphas": cfg_window.get("factors", {}).get("alphas"),
                    "target_vol": cfg_window.get("vol_target"),
                    "turnover_cap": cfg_window.get("turnover_cap"),
                },
            }
        )

        current_idx += _OUT_OF_SAMPLE_LEN

    combined_returns = combined_returns.sort_index()
    equity_curve = (1 + combined_returns).cumprod()

    if combined_weights:
        weights_df = pd.concat(combined_weights).fillna(0.0)
        turnover_mean = turnover(weights_df).mean()
    else:
        turnover_mean = np.nan

    kpis = {
        "CAGR": cagr(equity_curve),
        "Sharpe": sharpe(combined_returns),
        "Sortino": sortino(combined_returns),
        "Vol": vol(combined_returns),
        "MaxDD": mdd(equity_curve),
        "Calmar": calmar(equity_curve),
        "HitRate": hit_rate(combined_returns),
        "Turnover": float(turnover_mean) if not np.isnan(turnover_mean) else np.nan,
    }

    return {
        "equity_curve": equity_curve,
        "returns": combined_returns,
        "kpis": kpis,
        "windows": windows_info,
    }

