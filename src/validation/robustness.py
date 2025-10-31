from __future__ import annotations

"""Robustness and stress-testing utilities for backtests."""

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, vol
from reports.run_logging import append_run_log
from .walk_forward import run_walk_forward

__all__ = ["param_sensitivity_heatmaps", "stress_costs", "regime_subperiods"]

_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"

_PARAM_ALIASES = {
    "n_cubes": "mapper.n_cubes",
    "overlap": "mapper.overlap",
    "eps_quantile": "mapper.eps_quantile",
    "min_cluster_size": "mapper.min_cluster_size",
}

_ENV_MAP = {
    "factors.regime_gain": "REGIME_GAIN",
    "factors.regime_mode": "REGIME_MODE",
    "factors.softmax_T": "SOFTMAX_T",
}

_EMPTY_KPIS = {
    "CAGR": np.nan,
    "Sharpe": np.nan,
    "Sortino": np.nan,
    "Vol": np.nan,
    "MaxDD": np.nan,
    "Calmar": np.nan,
    "HitRate": np.nan,
}


def _timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _compute_kpis(
    equity: pd.Series,
    risk_free: Optional[pd.Series] = None,
) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    if isinstance(risk_free, pd.Series):
        rf_series = risk_free.reindex(returns.index).ffill().bfill().fillna(0.0)
    else:
        rf_series = None
    return {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(returns, risk_free=rf_series if rf_series is not None else 0.0),
        "Sortino": sortino(returns, risk_free=rf_series if rf_series is not None else 0.0),
        "Vol": vol(returns),
        "MaxDD": mdd(equity),
        "Calmar": calmar(equity),
        "HitRate": hit_rate(returns),
    }


def _assign_param(cfg: Dict, key: str, value: object) -> None:
    """Assign a parameter to the nested configuration, honouring legacy aliases."""
    normalized = _PARAM_ALIASES.get(key, key)
    if "." not in normalized and normalized in {"regime_gain", "regime_mode", "softmax_T"}:
        normalized = f"factors.{normalized}"
    path = normalized.split(".") if "." in normalized else [normalized]

    target = cfg
    for part in path[:-1]:
        target = target.setdefault(part, {})
    target[path[-1]] = value

    env_key = _ENV_MAP.get(normalized)
    if env_key:
        os.environ[env_key] = str(value)


def _extract_regime_stats(meta: Dict[str, object] | None) -> Dict[str, float]:
    if not isinstance(meta, dict):
        return {}
    tda_meta = meta.get("tda_params") if isinstance(meta.get("tda_params"), dict) else {}
    stats = meta.get("regime_stats") or meta.get("tfi_stats") or tda_meta.get("regime_stats")
    return stats if isinstance(stats, dict) else {}


def param_sensitivity_heatmaps(
    cfg: Dict,
    panel: pd.DataFrame,
    grid: Dict[str, Sequence[float]],
) -> List[Path]:
    """Generate Sharpe/Vol heatmaps for a 2D parameter grid."""

    if len(grid) != 2:
        raise ValueError("grid must contain exactly two parameters for heatmaps")

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    grid_values = [list(grid[k]) for k in keys]

    sharpe_matrix = np.zeros((len(grid_values[0]), len(grid_values[1])), dtype=float)
    vol_matrix = np.zeros_like(sharpe_matrix)
    runs_log: List[Dict[str, object]] = []

    for i, val_i in enumerate(grid_values[0]):
        for j, val_j in enumerate(grid_values[1]):
            cfg_run = deepcopy(cfg)
            _assign_param(cfg_run, keys[0], val_i)
            _assign_param(cfg_run, keys[1], val_j)

            try:
                result = run_walk_forward(cfg_run, panel=panel)
            except ValueError as err:
                print(
                    f"Walk-forward failed for {keys[0]}={val_i}, {keys[1]}={val_j}: {err}"
                )
                result = {"equity_curve": pd.Series(dtype=float), "windows": []}

            window_metas = [
                meta
                for meta in (window.get("meta") for window in result.get("windows", []))
                if isinstance(meta, dict)
            ]
            combined_meta = window_metas[-1] if window_metas else {}
            mapper_meta = (
                combined_meta.get("tda_params", {}).get("mapper_params", {})
                if isinstance(combined_meta.get("tda_params"), dict)
                else {}
            )
            regime_stats = _extract_regime_stats(combined_meta)

            equity_curve = result.get("equity_curve", pd.Series(dtype=float))
            risk_free_series = result.get("risk_free")
            kpis = (
                _compute_kpis(equity_curve, risk_free=risk_free_series)
                if not equity_curve.empty
                else dict(_EMPTY_KPIS)
            )

            sharpe_val = kpis.get("Sharpe", np.nan)
            vol_val = kpis.get("Vol", np.nan)
            sharpe_matrix[i, j] = sharpe_val
            vol_matrix[i, j] = vol_val

            factors_cfg = cfg_run.get("factors", {})
            mapper_cfg = cfg_run.get("mapper", {})
            tda_ph_cfg = cfg_run.get("tda_ph", {})

            runs_log.append(
                {
                    "i": i,
                    "j": j,
                    keys[0]: val_i,
                    keys[1]: val_j,
                    "factors.regime_gain": factors_cfg.get("regime_gain"),
                    "factors.regime_mode": factors_cfg.get("regime_mode"),
                    "factors.softmax_T": factors_cfg.get("softmax_T"),
                    "mapper_n_cubes": mapper_meta.get("n_cubes", mapper_cfg.get("n_cubes")),
                    "mapper_overlap": mapper_meta.get("overlap", mapper_cfg.get("overlap")),
                    "mapper_eps_quantile": mapper_meta.get("eps_quantile", mapper_cfg.get("eps_quantile")),
                    "mapper_min_cluster_size": mapper_meta.get("min_cluster_size", mapper_cfg.get("min_cluster_size")),
                    "tda_ph_window": tda_ph_cfg.get("window"),
                    "tda_ph_norm": tda_ph_cfg.get("norm"),
                    "regime_min": regime_stats.get("min"),
                    "regime_max": regime_stats.get("max"),
                    "regime_mean": regime_stats.get("mean"),
                    "regime_std": regime_stats.get("std"),
                    "sharpe": sharpe_val,
                    "vol": vol_val,
                    "cagr": kpis.get("CAGR"),
                    "max_drawdown": kpis.get("MaxDD"),
                    "calmar": kpis.get("Calmar"),
                    "hit_rate": kpis.get("HitRate"),
                }
            )

            append_run_log(
                cfg_run,
                metrics={
                    "sharpe": sharpe_val,
                    "vol": vol_val,
                    "cagr": kpis.get("CAGR"),
                    "max_drawdown": kpis.get("MaxDD"),
                    "calmar": kpis.get("Calmar"),
                    "hit_rate": kpis.get("HitRate"),
                },
                meta={
                    "source": "validation.robustness.param_sensitivity_heatmaps",
                    "grid_param_x": keys[0],
                    "grid_param_y": keys[1],
                    "grid_index_i": i,
                    "grid_index_j": j,
                },
            )

    df_sharpe = pd.DataFrame(
        sharpe_matrix,
        index=[f"{v:.3f}" for v in grid_values[0]],
        columns=[f"{v:.3f}" for v in grid_values[1]],
    )
    df_vol = pd.DataFrame(
        vol_matrix,
        index=[f"{v:.3f}" for v in grid_values[0]],
        columns=[f"{v:.3f}" for v in grid_values[1]],
    )
    df_sharpe.to_csv(_REPORT_DIR / "heatmap_sharpe_values.csv")
    df_vol.to_csv(_REPORT_DIR / "heatmap_vol_values.csv")
    pd.DataFrame(runs_log).to_csv(_REPORT_DIR / "heatmap_runs_log.csv", index=False)

    paths: List[Path] = []
    for matrix, metric_name in ((sharpe_matrix, "Sharpe"), (vol_matrix, "Vol")):
        fig, ax = plt.subplots(figsize=(8, 6))
        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin = vmax = 0.0
        if (vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(grid_values[1])))
        ax.set_xticklabels([f"{v:.2f}" for v in grid_values[1]])
        ax.set_yticks(range(len(grid_values[0])))
        ax.set_yticklabels([f"{v:.2f}" for v in grid_values[0]])
        ax.set_xlabel(keys[1])
        ax.set_ylabel(keys[0])
        ax.set_title(f"{metric_name} Sensitivity")
        fig.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()
        output_path = _REPORT_DIR / f"heatmap_{metric_name.lower()}_{_timestamp_tag()}.png"
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
        result = run_walk_forward(cfg_run, panel=panel)
        kpis = _compute_kpis(result["equity_curve"], risk_free=result.get("risk_free"))
        kpis.update({"multiplier": mult})
        records.append(kpis)

    df = pd.DataFrame(records)
    output_path = _REPORT_DIR / f"stress_costs_{_timestamp_tag()}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def regime_subperiods(
    cfg: Dict,
    panel: pd.DataFrame,
    regime_series: pd.Series,
    high_low_split: str | float = "median",
) -> pd.DataFrame:
    """Compute KPIs separately for high/low regime levels based on a scalar series."""

    if isinstance(high_low_split, str):
        if high_low_split != "median":
            raise ValueError("high_low_split string must be 'median'")
        threshold = regime_series.median()
    else:
        threshold = float(high_low_split)

    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    aligned_regime = regime_series.reindex(panel.index.get_level_values(date_level)).ffill().dropna()

    results: Dict[str, Dict[str, float]] = {}
    for label, mask in {
        "High": aligned_regime >= threshold,
        "Low": aligned_regime < threshold,
    }.items():
        selected_dates = mask[mask].index.unique()
        if selected_dates.empty:
            continue
        panel_subset = panel.loc[(selected_dates, slice(None)), :]
        cfg_run = deepcopy(cfg)
        cfg_run.setdefault("dates", {})
        cfg_run["dates"]["start"] = selected_dates.min().isoformat()
        cfg_run["dates"]["end"] = selected_dates.max().isoformat()
        try:
            result = run_walk_forward(cfg_run, panel=panel_subset)
        except ValueError:
            results[label] = dict(_EMPTY_KPIS)
            continue
        results[label] = _compute_kpis(result["equity_curve"], risk_free=result.get("risk_free"))

    return pd.DataFrame(results).T
