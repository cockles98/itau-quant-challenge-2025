from __future__ import annotations

"""Centralized risk guards for kill-switches and cooldown logic."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class KillSwitchResult:
    """Outcome for kill switch evaluation."""

    active: bool
    reason: str
    cooldown: int = 0


def rolling_mdd_kill_switch(
    equity: pd.Series,
    returns: pd.Series,
    target_vol: float,
    *,
    mdd_lookback: int,
    mdd_threshold: float,
    vol_multiplier: float,
) -> KillSwitchResult:
    if equity.empty:
        return KillSwitchResult(active=False, reason="insufficient_data")

    eq_tail = equity.tail(mdd_lookback).dropna()
    if len(eq_tail) >= 2:
        drawdown = eq_tail / eq_tail.cummax() - 1.0
        rolling_mdd = float(drawdown.min())
    else:
        rolling_mdd = 0.0

    vol_tail = returns.tail(mdd_lookback).dropna()
    if len(vol_tail) >= 2:
        realized_vol = float(vol_tail.std(ddof=0) * np.sqrt(252.0))
    else:
        realized_vol = 0.0

    vol_gate = realized_vol > (vol_multiplier * target_vol)
    mdd_gate = rolling_mdd <= mdd_threshold

    active = bool(vol_gate or mdd_gate)
    reason_parts = []
    if mdd_gate:
        reason_parts.append(f"mdd={rolling_mdd:.4f}")
    if vol_gate:
        reason_parts.append(f"vol={realized_vol:.4f}")
    reason = ",".join(reason_parts) if reason_parts else "ok"
    return KillSwitchResult(active=active, reason=reason)


def regime_cooldown_guard(
    regime_series: pd.Series,
    *,
    min_reset_days: int,
    alert_threshold: float,
    last_trigger_date: Optional[pd.Timestamp],
    current_date: pd.Timestamp,
) -> bool:
    if last_trigger_date is None:
        return False

    mask = (regime_series.index > last_trigger_date) & (regime_series.index <= current_date)
    window = regime_series.loc[mask]
    if window.empty:
        return True

    below_threshold = window < alert_threshold
    streak = int(below_threshold.tail(min_reset_days).sum())
    return streak >= min_reset_days
