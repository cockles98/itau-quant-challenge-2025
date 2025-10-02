from __future__ import annotations

"""Key performance indicator calculations for portfolio evaluation."""

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "sharpe",
    "sortino",
    "cagr",
    "vol",
    "mdd",
    "calmar",
    "turnover",
    "hit_rate",
    "avg_time_under_water",
    "max_time_under_water",
]

_DAYS_PER_YEAR = 252


def _to_series(data: Iterable[float]) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    return pd.Series(data).dropna()


def sharpe(
    returns: Iterable[float],
    risk_free: float = 0.0,
    periods_per_year: int = _DAYS_PER_YEAR,
) -> float:
    r = _to_series(returns)
    if r.empty:
        return np.nan
    excess = r - risk_free / periods_per_year
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() * periods_per_year / (std * np.sqrt(periods_per_year)))


def sortino(
    returns: Iterable[float],
    risk_free: float = 0.0,
    periods_per_year: int = _DAYS_PER_YEAR,
) -> float:
    r = _to_series(returns)
    if r.empty:
        return np.nan
    excess = r - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float(
        excess.mean() * periods_per_year / (downside_std * np.sqrt(periods_per_year))
    )


def cagr(equity: Iterable[float], periods_per_year: int = _DAYS_PER_YEAR) -> float:
    eq = _to_series(equity)
    if eq.empty:
        return np.nan
    start = eq.iloc[0]
    end = eq.iloc[-1]
    periods = len(eq)
    if start <= 0 or end <= 0 or periods <= 1:
        return np.nan
    years = periods / periods_per_year
    return float((end / start) ** (1.0 / years) - 1.0)


def vol(returns: Iterable[float], periods_per_year: int = _DAYS_PER_YEAR) -> float:
    r = _to_series(returns)
    if r.empty:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def mdd(equity: Iterable[float]) -> float:
    eq = _to_series(equity)
    if eq.empty:
        return np.nan
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    return float(drawdown.min())


def calmar(equity: Iterable[float], periods_per_year: int = _DAYS_PER_YEAR) -> float:
    growth = cagr(equity, periods_per_year)
    drawdown = mdd(equity)
    if drawdown is None or np.isnan(drawdown) or drawdown >= 0:
        return np.nan
    if np.isnan(growth):
        return np.nan
    return float(growth / abs(drawdown))


def turnover(weights: pd.DataFrame) -> pd.Series:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a DataFrame")
    aligned = weights.fillna(0.0)
    diffs = aligned.diff().abs()
    daily_turnover = 0.5 * diffs.sum(axis=1)
    return daily_turnover.dropna()


def hit_rate(returns: Iterable[float]) -> float:
    r = _to_series(returns)
    if r.empty:
        return np.nan
    positives = (r > 0).sum()
    total = (r != 0).sum()
    if total == 0:
        return np.nan
    return float(positives / total)

def _drawdown_durations(equity: Iterable[float]) -> list[int]:
    """Return lengths (in periods) of underwater episodes ending in a recovery."""

    eq = _to_series(equity)
    if eq.empty:
        return []

    running_max = eq.cummax()
    durations: list[int] = []
    stretch = 0
    for value, peak in zip(eq.values, running_max.values):
        if np.isnan(value) or np.isnan(peak):
            continue
        if value < peak - 1e-12:
            stretch += 1
        elif stretch > 0:
            durations.append(stretch)
            stretch = 0
    return durations


def avg_time_under_water(equity: Iterable[float]) -> float:
    """Average length of completed drawdown periods before recovering to the high-water mark."""

    durations = _drawdown_durations(equity)
    if not durations:
        return 0.0
    return float(np.mean(durations))


def max_time_under_water(equity: Iterable[float]) -> float:
    """Maximum length of completed drawdown periods before recovering to the high-water mark."""

    durations = _drawdown_durations(equity)
    if not durations:
        return 0.0
    return float(np.max(durations))

