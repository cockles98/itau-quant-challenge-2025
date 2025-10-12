from __future__ import annotations

"""Robustness and stress-testing utilities for backtests."""
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .walk_forward import run_walk_forward
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

    # sharpe_matrix = np.zeros((len(grid_values[0]), len(grid_values[1])))
    # vol_matrix = np.zeros_like(sharpe_matrix)
    sharpe_matrix = np.zeros((len(grid_values[0]), len(grid_values[1])), dtype=float)
    vol_matrix = np.zeros_like(sharpe_matrix)
    runs_log = []  # <-- novo: para CSV

    for i, val_i in enumerate(grid_values[0]):
        for j, val_j in enumerate(grid_values[1]):
            cfg_run = deepcopy(cfg)
            tda_cfg = cfg_run.setdefault("tda", {})
            fac_cfg = cfg_run.setdefault("factors", {})
            # tda_cfg[keys[1]] = val_j
            # tda_cfg[keys[0]] = val_i
            key_i, key_j = keys[0], keys[1]
            # --- aplica eixo X ---
            if key_i in ("regime_gain", "regime_mode"):
                if key_i == "regime_gain":
                    os.environ["REGIME_GAIN"] = str(val_i)
                else:
                    os.environ["REGIME_MODE"] = str(val_i)
            elif key_i == "softmax_T":
                fac_cfg["softmax_T"] = float(val_i)
                os.environ["SOFTMAX_T"] = str(val_i)
            else:
                tda_cfg[key_i] = val_i
            # --- aplica eixo Y ---
            if key_j in ("regime_gain", "regime_mode"):
                if key_j == "regime_gain":
                    os.environ["REGIME_GAIN"] = str(val_j)
                else:
                    os.environ["REGIME_MODE"] = str(val_j)
            elif key_j == "softmax_T":
                fac_cfg["softmax_T"] = float(val_j)
                os.environ["SOFTMAX_T"] = str(val_j)
            else:
                tda_cfg[key_j] = val_j
            # result = run_backtest(cfg_run, panel=panel)
            # LOG: prova de que os params chegaram
            try:
                result = run_walk_forward(cfg_run, panel=panel)
            except ValueError as err:
                print(
                    f"Walk-forward failed for {keys[0]}={val_i}, {keys[1]}={val_j}: {err}"
                )
                result = {
                    "equity_curve": pd.Series(dtype=float),
                    "windows": [],
                }
            window_metas = [
                meta
                for meta in (
                    window.get("meta") for window in result.get("windows", [])
                )
                if isinstance(meta, dict)
            ]
            combined_meta = window_metas[-1] if window_metas else {}
            used = combined_meta.get("tda_params") or (cfg_run.get("tda") or {})
            stats = combined_meta.get("tfi_stats") or {}
            runs_log.append({
                "i": i, "j": j,
                keys[0]: val_i, keys[1]: val_j,
                "regime_gain": os.getenv("REGIME_GAIN", None),
                "regime_mode": os.getenv("REGIME_MODE", None),
                "softmax_T": fac_cfg.get("softmax_T", None),
                "used_delay": used.get("delay"),
                "used_dim": used.get("dim"),
                "used_n_cubes": used.get("n_cubes"),
                "used_overlap": used.get("overlap"),
                "used_epsilon": used.get("epsilon"),
                "used_min_samples": used.get("min_samples"),
                "used_window": used.get("window"),
                # Range efetivo do TFI nesta execução (diagnóstico do “monocromático”)
                "tfi_min": stats.get("min"),
                "tfi_max": stats.get("max"),
                "tfi_std": stats.get("std"),
                "tfi_mean": stats.get("mean"),
            })
            equity_curve = result.get("equity_curve", pd.Series(dtype=float))
            returns = equity_curve.pct_change().dropna()
            sharpe_matrix[i, j] = sharpe(returns) if not returns.empty else np.nan
            vol_matrix[i, j] = vol(returns) if not returns.empty else np.nan

    # Salva as matrizes e o log (debug duro)
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

    paths = []
    for matrix, metric_name in ((sharpe_matrix, "Sharpe"), (vol_matrix, "Vol")):
        fig, ax = plt.subplots(figsize=(8, 6))
        # im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        # Fixar vmin/vmax para não "equalizar" cada heatmap isoladamente
        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
        # Se a matriz for praticamente constante, garanta um range > 0 para o colormap
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 0.0
        if (vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
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
        result = run_walk_forward(cfg_run, panel=panel)
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
        try:
            result = run_walk_forward(cfg_run, panel=panel_subset)
        except ValueError:
            results[label] = {key: float("nan") for key in ("CAGR", "Sharpe", "Sortino", "Vol", "MaxDD", "Calmar", "HitRate")}
            continue
        results[label] = _compute_kpis(result["equity_curve"])

    return pd.DataFrame(results).T
