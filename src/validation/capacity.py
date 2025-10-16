from __future__ import annotations

"""Capacity analysis utilities."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .walk_forward import run_walk_forward
from metrics import mdd, sharpe

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
        risk_section = cfg_run.setdefault("risk", {})
        if not isinstance(risk_section, dict):
            risk_section = {}
            cfg_run["risk"] = risk_section
        risk_section["participation_cap"] = cap
        result = run_walk_forward(cfg_run, panel=panel)
        equity = result["equity_curve"]
        returns = equity.pct_change().dropna()
        kpis = result.get("kpis", {}) or {}
        turn = float(kpis.get("Turnover", float("nan")))
        window_metas = [
            meta
            for meta in (window.get("meta") for window in result.get("windows", []))
            if isinstance(meta, dict)
        ]
        cap_rates: List[float] = []
        cut_fracs: List[float] = []
        for meta in window_metas:
            cap_meta = meta.get("capacity") or {}
            cap_rates.append(float(cap_meta.get("cap_bind_rate", np.nan)))
            cut_fracs.append(float(cap_meta.get("avg_turnover_cut_frac", np.nan)))
        cap_bind_rate = float(np.nanmean(cap_rates)) if cap_rates else float("nan")
        avg_cut = float(np.nanmean(cut_fracs)) if cut_fracs else float("nan")
        records.append(
            {
                "participation_cap": cap,
                "Sharpe": sharpe(returns),
                "MaxDD": mdd(equity),  # negativo por convenção
                "Turnover": turn,
                "cap_bind_rate": cap_bind_rate,
                "avg_turnover_cut_frac": avg_cut,
            }
        )

    df = pd.DataFrame(records)
    # Para visualização intuitiva, use DD positivo no gráfico:
    df["MaxDD_pos"] = -df["MaxDD"]

    timestamp = _timestamp_tag()
    csv_path = _REPORT_DIR / f"capacity_curve_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        df["participation_cap"],
        df["Sharpe"],
        marker="o",
        color="tab:blue",
        label="Sharpe",
    )
    ax1.set_xlabel("Participation Cap")
    ax1.set_ylabel("Sharpe", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        df["participation_cap"], df["MaxDD_pos"], marker="s", color="tab:red", label="MaxDD"
    )
    ax2.set_ylabel("Max Drawdown (positivo, 0.49 = 49%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    # anota binding rate (se disponível)
    if "cap_bind_rate" in df.columns and df["cap_bind_rate"].notna().any():
        for x, y, r in zip(df["participation_cap"], df["MaxDD_pos"], df["cap_bind_rate"]):
            if np.isfinite(r):
                ax2.annotate(f"{r:.1%}", (x, y), textcoords="offset points",
                             xytext=(0, -15), ha="center", fontsize=8, color="tab:red")

    plt.title("Capacity vs Performance")
    plt.tight_layout()
    fig_path = _REPORT_DIR / f"capacity_curve_{timestamp}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    return {"csv": csv_path, "figure": fig_path}
