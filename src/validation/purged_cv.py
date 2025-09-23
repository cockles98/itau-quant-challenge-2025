from __future__ import annotations

"""Purged K-fold cross-validation utilities for time series."""

from copy import deepcopy
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from metrics import sharpe

__all__ = ["purged_kfold_split", "tune_params"]


def purged_kfold_split(
    dates: Sequence[pd.Timestamp] | pd.Index,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-fold splits with temporal embargo to avoid leakage."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")

    dates_index = pd.Index(dates)
    if not dates_index.is_monotonic_increasing:
        dates_index = dates_index.sort_values()

    fold_sizes = np.full(n_splits, len(dates_index) // n_splits, dtype=int)
    fold_sizes[: len(dates_index) % n_splits] += 1

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_indices = np.arange(start, stop)

        embargo_start = max(0, start - embargo_days)
        embargo_end = min(len(dates_index), stop + embargo_days)
        train_indices = np.concatenate(
            (np.arange(0, embargo_start), np.arange(embargo_end, len(dates_index)))
        )

        folds.append((train_indices, test_indices))
        start = stop

    return folds


def tune_params(
    cfg: Dict,
    panel: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    n_splits: int = 5,
    embargo_days: int = 5,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Grid-search hyperparameters using purged K-fold Sharpe OOS."""

    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    unique_dates = panel.index.get_level_values(date_level).unique().sort_values()
    folds = purged_kfold_split(
        unique_dates, n_splits=n_splits, embargo_days=embargo_days
    )

    results = []
    best_score = -np.inf
    best_params = None

    grid_keys = list(param_grid.keys())
    grid_values = [list(param_grid[k]) for k in grid_keys]

    for combo in product(*grid_values):
        params = dict(zip(grid_keys, combo))
        sharpe_scores = []
        for _, test_idx in folds:
            cfg_run = deepcopy(cfg)
            cfg_run.setdefault("dates", {})
            cfg_run["dates"]["start"] = unique_dates.min().isoformat()
            cfg_run["dates"]["end"] = unique_dates.max().isoformat()

            factors_cfg = cfg_run.setdefault("factors", {})
            factors_cfg["alphas"] = [
                params.get("alpha", 0.6),
                params.get("beta", 0.3),
                params.get("gamma", 0.1),
            ]

            tda_cfg = cfg_run.setdefault("tda", {})
            for key in ("delay", "dim", "n_cubes", "overlap"):
                if key in params:
                    tda_cfg[key] = params[key]

            result = run_backtest(cfg_run, panel=panel)
            equity = result["equity_curve"]
            test_dates = unique_dates[test_idx]
            equity_subset = equity.loc[test_dates.min() : test_dates.max()]
            returns = equity_subset.pct_change().dropna()
            sharpe_scores.append(sharpe(returns))

        avg_sharpe = np.nanmean(sharpe_scores)
        results.append({"params": params, "sharpe": avg_sharpe})
        if avg_sharpe > best_score:
            best_score = avg_sharpe
            best_params = params

    results_df = pd.DataFrame(results)
    return best_params or {}, results_df
