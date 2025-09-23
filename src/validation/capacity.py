from __future__ import annotations

"""Capacity analysis utilities."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from backtest.engine import run_backtest
from metrics import mdd, sharpe, turnover

__all__ = ["capacity_curve"]

_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def _timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def capacity_curve(
    cfg: Dict,
    panel: pd.DataFrame,
    participation_caps: Iterable[float] = (0.01, 0.02, 0.05, 0.10),
) -> Dict[str, Path]:
    """Evaluate Sharpe/MDD/turnover sensitivity to participation caps."""

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for cap in participation_caps:
        cfg_run = deepcopy(cfg)
        cfg_run["participation_cap"] = cap
        result = run_backtest(cfg_run, panel=panel)
        equity = result["equity_curve"]
        returns = equity.pct_change().dropna()
        weights = result.get("daily_positions")
        turn = turnover(weights).mean() if isinstance(weights, pd.DataFrame) else float("nan")
        records.append(
            {
                "participation_cap": cap,
                "Sharpe": sharpe(returns),
                "MaxDD": mdd(equity),
                "Turnover": turn,
            }
        )

    df = pd.DataFrame(records)
    timestamp = _timestamp_tag()
    csv_path = _REPORT_DIR / f"capacity_curve_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["participation_cap"], df["Sharpe"], marker="o", color="tab:blue", label="Sharpe")
    ax1.set_xlabel("Participation Cap")
    ax1.set_ylabel("Sharpe", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df["participation_cap"], df["MaxDD"], marker="s", color="tab:red", label="MaxDD")
    ax2.set_ylabel("Max Drawdown", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Capacity vs Performance")
    plt.tight_layout()
    fig_path = _REPORT_DIR / f"capacity_curve_{timestamp}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    return {"csv": csv_path, "figure": fig_path}
