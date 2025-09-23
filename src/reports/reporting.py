from __future__ import annotations

"""Reporting utilities for visualising backtest results."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, turnover, vol

_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

__all__ = ["plot_equity_curves", "table_kpis"]


def plot_equity_curves(curves: Dict[str, pd.Series], filename: str = "equity_curves.png") -> Path:
    if not curves:
        raise ValueError("curves dictionary is empty")

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for label, series in curves.items():
        if not isinstance(series, pd.Series):
            raise TypeError("Each curve must be a pandas Series")
        ax.plot(series.index, series.values, label=label)
    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = _REPORTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def table_kpis(results: Dict[str, Dict[str, object]], filename: str = "kpi_table.csv") -> pd.DataFrame:
    if not results:
        raise ValueError("results dictionary is empty")

    rows = {}
    for label, payload in results.items():
        equity = payload.get("equity_curve")
        rets = payload.get("returns")
        weights = payload.get("weights")
        if equity is None or rets is None:
            raise ValueError(f"Missing equity_curve or returns for {label}")

        row = {
            "CAGR": cagr(equity),
            "Sharpe": sharpe(rets),
            "Sortino": sortino(rets),
            "Vol": vol(rets),
            "MaxDD": mdd(equity),
            "Calmar": calmar(equity),
            "HitRate": hit_rate(rets),
            "Turnover": turnover(weights).mean() if isinstance(weights, pd.DataFrame) else pd.NA,
        }
        rows[label] = row

    table = pd.DataFrame(rows).T
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _REPORTS_DIR / filename
    table.to_csv(output_path)
    return table
