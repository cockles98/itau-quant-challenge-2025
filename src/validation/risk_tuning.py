from __future__ import annotations

"""Utilities to tune risk configuration hyperparameters via brute-force search."""

from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, vol


__all__ = ["tune_risk_parameters", "risk_heatmap"]

_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "risk_tuning"


def _timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _set_nested(mapping: MutableMapping, dotted_key: str, value) -> None:
    """Assign ``value`` in ``mapping`` following a dotted path (e.g. ``risk.mdd_thres``)."""

    parts = dotted_key.split(".")
    node: MutableMapping = mapping
    for key in parts[:-1]:
        child = node.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            node[key] = child
        node = child
    node[parts[-1]] = value


def _compute_metrics(equity: pd.Series) -> Dict[str, float]:
    """Return key performance indicators used during tuning."""

    returns = equity.pct_change().dropna()
    return {
        "cagr": cagr(equity),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "vol": vol(returns),
        "max_drawdown": mdd(equity),
        "calmar": calmar(equity),
        "hit_rate": hit_rate(returns),
    }


def _cartesian_product(param_grid: Mapping[str, Sequence]) -> List[Tuple]:
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    combos = list(product(*values))
    return keys, combos


def tune_risk_parameters(
    cfg: Dict,
    panel: pd.DataFrame,
    param_grid: Mapping[str, Sequence],
    *,
    metrics: Optional[Iterable[str]] = None,
    output_dir: Optional[Path | str] = None,
    save_csv: bool = True,
    isolate_artifacts: bool = False,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Run a brute-force sweep over the provided ``param_grid`` for ``cfg['risk']``.

    Parameters
    ----------
    cfg:
        Base configuration dictionary. The object is not modified in-place.
    panel:
        Market data already loaded (MultiIndex by date/asset).
    param_grid:
        Mapping from dotted keys to iterables of candidate values. The keys are
        expected to live under ``risk`` (e.g. ``risk.target_vol``), but any dotted
        path is accepted.
    metrics:
        Subset of KPI names to keep in the final DataFrame. Defaults to the full set.
    output_dir:
        Optional directory to save the CSV with raw results. Defaults to
        ``reports/risk_tuning``.
    save_csv:
        Persist results to CSV when True (default). Returns the path alongside
        the DataFrame. When False, the second element of the tuple is ``None``.
    isolate_artifacts:
        When True, each configuration is executed with its own artifact/report
        directory to avoid cache reuse. Useful for testing or when caches are
        not compatible across grid points.
    """

    if not isinstance(param_grid, Mapping) or not param_grid:
        raise ValueError("param_grid must be a non-empty mapping")

    metric_subset = list(metrics) if metrics is not None else None

    keys, combos = _cartesian_product(param_grid)
    if not combos:
        raise ValueError("param_grid produced no combinations")

    results: List[Dict[str, float]] = []
    for idx, values in enumerate(combos):
        cfg_run = deepcopy(cfg)
        paths_cfg = cfg_run.setdefault("paths", {})
        if isolate_artifacts:
            base_artifacts = Path(paths_cfg.get("artifacts", _REPORT_DIR.parents[0] / "artifacts"))
            unique_root = base_artifacts / f"risk_tuning_{idx:04d}"
            paths_cfg["artifacts"] = str(unique_root)
            paths_cfg["reports"] = str(unique_root / "reports")
        for key, value in zip(keys, values):
            _set_nested(cfg_run, key, value)
        result = run_backtest(cfg_run, panel=panel)
        metrics_dict = _compute_metrics(result["equity_curve"])
        row = {key: value for key, value in zip(keys, values)}
        row.update(metrics_dict)
        results.append(row)

    df = pd.DataFrame(results)
    if metric_subset:
        missing = sorted(set(metric_subset) - set(df.columns))
        if missing:
            raise ValueError(f"Unknown metric(s) requested: {missing}")
        df = df[[*keys, *metric_subset]]

    csv_path: Optional[Path] = None
    if save_csv:
        target_dir = Path(output_dir) if output_dir else _REPORT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"risk_grid_{_timestamp_tag()}.csv"
        df.to_csv(csv_path, index=False)

    return df, csv_path


def risk_heatmap(
    cfg: Dict,
    panel: pd.DataFrame,
    param_grid: Mapping[str, Sequence],
    *,
    metric: str = "sharpe",
    output_dir: Optional[Path | str] = None,
    isolate_artifacts: bool = False,
) -> Tuple[pd.DataFrame, Path]:
    """Convenience helper to visualise 2-D risk sweeps as heatmaps."""

    if len(param_grid) != 2:
        raise ValueError("risk_heatmap expects exactly two parameters")

    df, csv_path = tune_risk_parameters(
        cfg,
        panel,
        param_grid,
        metrics=[metric],
        output_dir=output_dir,
        save_csv=True,
        isolate_artifacts=isolate_artifacts,
    )
    if csv_path is None:
        raise RuntimeError("CSV path unexpectedly missing after tuning run")

    keys = list(param_grid.keys())
    pivot = df.pivot(index=keys[0], columns=keys[1], values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    matrix = pivot.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-9
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    ax.set_xlabel(keys[1])
    ax.set_ylabel(keys[0])
    ax.set_title(f"{metric.title()} heatmap")
    fig.colorbar(im, ax=ax, label=metric.title())
    fig.tight_layout()

    target_dir = Path(output_dir) if output_dir else _REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = target_dir / f"risk_heatmap_{metric}_{_timestamp_tag()}.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    return df, heatmap_path
