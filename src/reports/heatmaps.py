from __future__ import annotations

"""Heatmap utilities for parameter sensitivity sweeps."""

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reports.run_logging import append_run_log
from validation.walk_forward import run_walk_forward

__all__ = ["run_two_param_heatmaps", "run_param_grid"]

_DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "parameter_sweeps"


def _set_nested(cfg_dict: Dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = cfg_dict
    for key in parts[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[parts[-1]] = value


def _extract_metrics(kpis: Mapping[str, float] | None) -> Dict[str, float]:
    if not isinstance(kpis, Mapping):
        return {
            "cagr": np.nan,
            "sharpe": np.nan,
            "vol": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }
    cagr = kpis.get("CAGR", kpis.get("annual_return"))
    sharpe_val = kpis.get("Sharpe", kpis.get("sharpe"))
    vol = kpis.get("Vol", kpis.get("annual_vol"))
    maxdd = kpis.get("MaxDD", kpis.get("max_drawdown"))
    calmar = np.nan
    if maxdd not in (None, 0) and cagr not in (None,):
        if np.isfinite(maxdd) and np.isfinite(cagr or np.nan):
            denom = abs(maxdd)
            calmar = cagr / denom if denom > 0 else np.nan
    return {
        "cagr": cagr,
        "sharpe": sharpe_val,
        "vol": vol,
        "max_drawdown": maxdd,
        "calmar": calmar,
    }


def run_two_param_heatmaps(
    cfg_base: Mapping,
    panel_data: pd.DataFrame,
    grid: Mapping[str, Sequence],
    report_prefix: str,
    *,
    plot_metrics: Iterable[str] = ("sharpe", "max_drawdown"),
    output_dir: Path | str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    if len(grid) != 2:
        raise ValueError("Grid precisa ter exatamente dois parametros para heatmap.")

    axes = list(grid.keys())
    axis_values = [list(grid[axes[0]]), list(grid[axes[1]])]
    metric_surfaces = {
        metric: np.full((len(axis_values[0]), len(axis_values[1])), np.nan)
        for metric in plot_metrics
    }
    records = []
    out_dir = Path(output_dir) if output_dir else _DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, val_x in enumerate(axis_values[0]):
        for j, val_y in enumerate(axis_values[1]):
            cfg_run = deepcopy(cfg_base)
            _set_nested(cfg_run, axes[0], val_x)
            _set_nested(cfg_run, axes[1], val_y)
            print(f" - Walk-forward {report_prefix} | {axes[0]}={val_x}, {axes[1]}={val_y}")
            result = run_walk_forward(cfg_run, panel=panel_data)
            metrics = _extract_metrics(result.get("kpis", {}))
            append_run_log(
                cfg_run,
                metrics=metrics,
                meta={
                    "source": "reports.heatmaps.run_two_param_heatmaps",
                    "report_prefix": report_prefix,
                    "grid_param_x": axes[0],
                    "grid_param_y": axes[1],
                    "grid_index_i": i,
                    "grid_index_j": j,
                },
            )
            record = {axes[0]: val_x, axes[1]: val_y, **metrics}
            records.append(record)
            for metric in plot_metrics:
                metric_surfaces[metric][i, j] = metrics.get(metric)

    results_df = pd.DataFrame(records)
    csv_path = out_dir / f"{report_prefix}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Resultados salvos em {csv_path}")

    heatmap_paths: Dict[str, Path] = {}
    for metric, surface in metric_surfaces.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(surface, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(axis_values[1])))
        ax.set_xticklabels([str(v) for v in axis_values[1]])
        ax.set_yticks(range(len(axis_values[0])))
        ax.set_yticklabels([str(v) for v in axis_values[0]])
        ax.set_xlabel(axes[1])
        ax.set_ylabel(axes[0])
        ax.set_title(f"{metric.title()} - {report_prefix}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path = out_dir / f"{report_prefix}_{metric}_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        heatmap_paths[metric] = out_path

    return results_df, heatmap_paths


def run_param_grid(
    cfg_base: Mapping,
    panel_data: pd.DataFrame,
    grid: Mapping[str, Sequence],
    report_prefix: str,
    *,
    output_dir: Path | str | None = None,
) -> Tuple[pd.DataFrame, Path]:
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("grid must be a non-empty mapping")

    out_dir = Path(output_dir) if output_dir else _DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    values = [list(grid[key]) for key in keys]
    records = []
    for combo in product(*values):
        cfg_run = deepcopy(cfg_base)
        for key, value in zip(keys, combo):
            _set_nested(cfg_run, key, value)
        result = run_walk_forward(cfg_run, panel=panel_data)
        metrics = _extract_metrics(result.get("kpis", {}))
        append_run_log(
            cfg_run,
            metrics=metrics,
            meta={
                "source": "reports.heatmaps.run_param_grid",
                "report_prefix": report_prefix,
            },
        )
        record = {key: value for key, value in zip(keys, combo)}
        record.update(metrics)
        records.append(record)

    df = pd.DataFrame(records)
    csv_path = out_dir / f"{report_prefix}_grid.csv"
    df.to_csv(csv_path, index=False)
    print(f"Resultados salvos em {csv_path}")
    return df, csv_path
