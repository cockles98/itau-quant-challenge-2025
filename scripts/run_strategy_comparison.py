
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from dataio.config import load_config
from dataio.loaders import get_panel


def _ensure_matching_dates(base_cfg: dict, other_cfg: dict) -> dict:
    other = deepcopy(other_cfg)
    base_dates = base_cfg.get("dates", {})
    other.setdefault("dates", {})
    other["dates"]["start"] = base_dates.get("start")
    other["dates"]["end"] = base_dates.get("end")
    return other


def _resolve_path(base_cfg_path: str, relative: str) -> Path:
    base_path = Path(base_cfg_path).resolve()
    root = base_path.parent
    if root.name == "configs":
        root = root.parent
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def run_comparison(base_cfg_path: str, hrp_cfg_path: str, tda_cfg_path: str) -> None:
    base_cfg = load_config(base_cfg_path)
    hrp_cfg = _ensure_matching_dates(base_cfg, load_config(hrp_cfg_path))
    tda_cfg = _ensure_matching_dates(base_cfg, load_config(tda_cfg_path))

    dates_cfg = base_cfg.get("dates", {})
    start = dates_cfg.get("start")
    end = dates_cfg.get("end")
    if not start or not end:
        raise ValueError("Base configuration must define dates.start and dates.end")

    panel = get_panel(start, end)

    reports_dir = _resolve_path(base_cfg_path, base_cfg.get("paths", {}).get("reports", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("strategy", base_cfg),
        ("hrp_only", hrp_cfg),
        ("tda_only", tda_cfg),
    ]

    comparison = []
    curves = {}
    for label, raw_cfg in variants:
        cfg = deepcopy(raw_cfg)
        result = run_backtest(cfg, panel=panel)
        equity = result["equity_curve"].rename(label)
        curves[label] = equity
        kpi_row = {"strategy": label}
        kpi_row.update(result.get("kpis", {}))
        comparison.append(kpi_row)

    # Ibovespa benchmark
    data_dir = _resolve_path(base_cfg_path, base_cfg.get("paths", {}).get("data", "./data"))
    ibov_path = data_dir / "benchmark" / "ibovespa_index_b3.csv"
    if ibov_path.exists():
        ibov_series = (
            pd.read_csv(ibov_path, parse_dates=["date"], index_col="date")["close"]
            .sort_index()
        )
        strategy_index = curves["strategy"].index
        ibov_aligned = ibov_series.reindex(strategy_index).ffill()
        if not ibov_aligned.isna().all():
            ibov_equity = (ibov_aligned / ibov_aligned.iloc[0]).rename("ibovespa")
            ibov_returns = ibov_equity.pct_change().fillna(0.0)
            curves["ibovespa"] = ibov_equity

            periods = len(ibov_returns)
            ann_return = (
                float(ibov_equity.iloc[-1] ** (252 / periods) - 1.0)
                if periods > 0 and ibov_equity.iloc[0] > 0
                else float("nan")
            )
            ann_vol = float(ibov_returns.std(ddof=0) * np.sqrt(252)) if periods > 1 else float("nan")
            sharpe = (
                ann_return / ann_vol
                if ann_vol and ann_vol > 0 and not np.isnan(ann_return)
                else float("nan")
            )
            comparison.append(
                {
                    "strategy": "ibovespa",
                    "final_equity": float(ibov_equity.iloc[-1]),
                    "total_return": float(ibov_equity.iloc[-1] - 1.0),
                    "annual_return": ann_return,
                    "annual_vol": ann_vol,
                    "sharpe": float(sharpe),
                    "max_drawdown": float((ibov_equity / ibov_equity.cummax() - 1.0).min()),
                    "avg_time_under_water": float("nan"),
                    "max_time_under_water": float("nan"),
                }
            )
        else:
            print("Ibovespa benchmark has no overlap with strategy dates; skipping curve.")
    else:
        print(f"Ibovespa benchmark file not found at {ibov_path}; skipping curve.")

    equity_df = pd.DataFrame(curves)
    equity_path = reports_dir / "strategy_comparison_equity.csv"
    equity_df.to_csv(equity_path, index_label="date")

    kpi_df = pd.DataFrame(comparison)
    kpi_path = reports_dir / "strategy_comparison_kpis.csv"
    kpi_df.to_csv(kpi_path, index=False)

    print(f"Saved equity curves to {equity_path}")
    print(f"Saved KPI comparison to {kpi_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base strategy with HRP-only and TDA-only variants",
    )
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to primary strategy YAML",
    )
    parser.add_argument(
        "--hrp-config",
        default="configs/hrp_only.yaml",
        help="Path to HRP-only YAML",
    )
    parser.add_argument(
        "--tda-config",
        default="configs/tda_only.yaml",
        help="Path to TDA-only YAML",
    )
    args = parser.parse_args()
    run_comparison(args.base_config, args.hrp_config, args.tda_config)


if __name__ == "__main__":
    main()

