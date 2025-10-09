#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from dataio.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PH-based regime threshold as a risk filter."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Configuration YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
        help="Z-score thresholds (in sigma) for the risk filter (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="reports/ph_threshold_backtest",
        help="Output stem for CSV/JSON reports (default: %(default)s)",
    )
    return parser.parse_args()


def _load_regime_series(meta: dict) -> pd.Series:
    topology_meta = meta.get("tda_topology", {})
    csv_path = topology_meta.get("regime_report")
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        if "regime" in df.columns:
            return df["regime"].astype(float)
    stats = topology_meta.get("regime_stats", {})
    mean = float(stats.get("mean", 0.0))
    index = pd.DatetimeIndex(meta.get("regime_controls", {}).get("per_date", []))
    if index.empty:
        index = pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.Series(mean, index=index, dtype=float)


def _compute_equity(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity


def _compute_metrics(equity: pd.Series) -> dict:
    if equity.empty:
        return {"final_equity": 1.0}
    returns = equity.pct_change().fillna(0.0)
    ann_return = equity.iloc[-1] ** (252 / len(equity)) - 1.0 if len(equity) > 0 else np.nan
    ann_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 1e-9 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "final_equity": float(equity.iloc[-1]),
        "annual_return": float(ann_return),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
    }


def _apply_filter(base_returns: pd.Series, regime_z: pd.Series, threshold: float) -> pd.Series:
    aligned = base_returns.reindex(regime_z.index).fillna(0.0)
    mask = regime_z <= threshold
    filtered = aligned.where(mask, 0.0)
    return filtered


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    print("[INFO] Running baseline backtest...")
    baseline = run_backtest(cfg)
    equity_curve = baseline["equity_curve"]
    base_returns = equity_curve.pct_change().fillna(0.0)
    meta = baseline.get("meta", {})
    regime_series = _load_regime_series(meta).reindex(base_returns.index).fillna(method="ffill").fillna(0.0)
    regime_z = _zscore(regime_series)

    results = []
    base_metrics = {
        "threshold": None,
        "final_equity": float(equity_curve.iloc[-1]),
        "annual_return": float(baseline["kpis"].get("annual_return", np.nan)),
        "annual_vol": float(baseline["kpis"].get("annual_vol", np.nan)),
        "sharpe": float(baseline["kpis"].get("sharpe", np.nan)),
        "max_drawdown": float(baseline["kpis"].get("max_drawdown", np.nan)),
        "calmar": float(
            baseline["kpis"].get("annual_return", np.nan)
            / abs(baseline["kpis"].get("max_drawdown", -np.nan))
            if baseline["kpis"].get("max_drawdown", 0) < 0
            else np.nan
        ),
        "exposure_ratio": 1.0,
    }
    results.append(base_metrics)

    for threshold in args.thresholds:
        filtered_returns = _apply_filter(base_returns, regime_z, threshold)
        exposure_ratio = (regime_z <= threshold).mean()
        equity = _compute_equity(filtered_returns)
        metrics = _compute_metrics(equity)
        metrics.update(
            {
                "threshold": threshold,
                "exposure_ratio": float(exposure_ratio),
            }
        )
        results.append(metrics)
        print(f"[INFO] Threshold {threshold:.2f}: final equity={metrics['final_equity']:.3f}, sharpe={metrics['sharpe']:.3f}")

    df = pd.DataFrame(results)
    csv_path, json_path = _write_outputs(args.output, df, regime_series, regime_z)
    print(f"[INFO] Wrote summary CSV to {csv_path}")
    print(f"[INFO] Wrote detail JSON to {json_path}")


def _write_outputs(stem: str, df: pd.DataFrame, regime: pd.Series, regime_z: pd.Series) -> Tuple[Path, Path]:
    stem_path = Path(stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem_path.with_suffix(".csv")
    json_path = stem_path.with_suffix(".json")

    df.to_csv(csv_path, index=False)
    payload = {
        "results": df.to_dict(orient="records"),
        "regime_summary": {
            "min": float(regime.min()),
            "max": float(regime.max()),
            "mean": float(regime.mean()),
            "std": float(regime.std(ddof=0)),
        },
        "regime_z_summary": {
            "min": float(regime_z.min()),
            "max": float(regime_z.max()),
            "mean": float(regime_z.mean()),
            "std": float(regime_z.std(ddof=0)),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


if __name__ == "__main__":
    main()

