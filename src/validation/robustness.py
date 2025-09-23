from __future__ import annotations

"""Robustness and stress-testing utilities for backtests."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, vol

__all__ = ["param_sensitivity_heatmaps", "stress_costs", "regime_subperiods"]

_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def _timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _compute_kpis(equity: pd.Series) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    return {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "Vol": vol(returns),
        "MaxDD": mdd(equity),
        "Calmar": calmar(equity),
        "HitRate": hit_rate(returns),
    }


def param_sensitivity_heatmaps(
    cfg: Dict,
    panel: pd.DataFrame,
    grid: Dict[str, Sequence[float]],
) -> List[Path]:
    """Generate Sharpe/Vol heatmaps for a 2D parameter grid.

    Parameters
    ----------
    cfg : dict
        Base backtest configuration.
    panel : DataFrame
        Market data already loaded (MultiIndex by date/asset).
    grid : dict
        Dictionary with exactly two entries defining the parameter axes.
    """

    if len(grid) != 2:
        raise ValueError("grid must contain exactly two parameters for heatmaps")

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(grid.keys())
    grid_values = [list(grid[k]) for k in keys]

    sharpe_matrix = np.zeros((len(grid_values[0]), len(grid_values[1])))
    vol_matrix = np.zeros_like(sharpe_matrix)

    for i, val_i in enumerate(grid_values[0]):
        for j, val_j in enumerate(grid_values[1]):
            cfg_run = deepcopy(cfg)
            tda_cfg = cfg_run.setdefault("tda", {})
            tda_cfg[keys[0]] = val_i
            tda_cfg[keys[1]] = val_j
            result = run_backtest(cfg_run, panel=panel)
            returns = result["equity_curve"].pct_change().dropna()
            sharpe_matrix[i, j] = sharpe(returns)
            vol_matrix[i, j] = vol(returns)

    paths = []
    for matrix, metric_name in ((sharpe_matrix, "Sharpe"), (vol_matrix, "Vol")):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(grid_values[1])))
        ax.set_xticklabels([f"{v:.2f}" for v in grid_values[1]])
        ax.set_yticks(range(len(grid_values[0])))
        ax.set_yticklabels([f"{v:.2f}" for v in grid_values[0]])
        ax.set_xlabel(keys[1])
        ax.set_ylabel(keys[0])
        ax.set_title(f"{metric_name} Sensitivity")
        fig.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()
        output_path = (
            _REPORT_DIR / f"heatmap_{metric_name.lower()}_{_timestamp_tag()}.png"
        )
        fig.savefig(output_path)
        plt.close(fig)
        paths.append(output_path)

    return paths


def stress_costs(
    cfg: Dict,
    panel: pd.DataFrame,
    multipliers: Iterable[float] = (0.5, 1.0, 2.0),
) -> Path:
    """Stress test transaction cost assumptions by scaling fee/slippage."""

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base_costs = deepcopy(cfg.get("costs", {}))
    records = []
    for mult in multipliers:
        cfg_run = deepcopy(cfg)
        cost_cfg = cfg_run.setdefault("costs", {})
        for key, value in base_costs.items():
            cost_cfg[key] = value * mult
        result = run_backtest(cfg_run, panel=panel)
        kpis = _compute_kpis(result["equity_curve"])
        kpis.update({"multiplier": mult})
        records.append(kpis)

    df = pd.DataFrame(records)
    output_path = _REPORT_DIR / f"stress_costs_{_timestamp_tag()}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def regime_subperiods(
    cfg: Dict,
    panel: pd.DataFrame,
    tfi: pd.Series,
    high_low_split: str | float = "median",
) -> pd.DataFrame:
    """Compute KPIs separately for high/low TFI regimes."""

    if isinstance(high_low_split, str):
        if high_low_split != "median":
            raise ValueError("high_low_split string must be 'median'")
        threshold = tfi.median()
    else:
        threshold = float(high_low_split)

    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    tfi_aligned = tfi.reindex(panel.index.get_level_values(date_level)).ffill().dropna()

    results = {}
    for label, mask in {
        "High": tfi_aligned >= threshold,
        "Low": tfi_aligned < threshold,
    }.items():
        selected_dates = mask[mask].index.unique()
        if selected_dates.empty:
            continue
        panel_subset = panel.loc[(selected_dates, slice(None)), :]
        cfg_run = deepcopy(cfg)
        cfg_run.setdefault("dates", {})
        cfg_run["dates"]["start"] = selected_dates.min().isoformat()
        cfg_run["dates"]["end"] = selected_dates.max().isoformat()
        result = run_backtest(cfg_run, panel=panel_subset)
        results[label] = _compute_kpis(result["equity_curve"])

    return pd.DataFrame(results).T
