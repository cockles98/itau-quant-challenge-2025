from __future__ import annotations

"""
Statistical validation helpers (probabilistic/deflated Sharpe, permutation tests).
"""

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from metrics import sharpe

_DAYS_PER_YEAR = 252


def _to_series(data: Iterable[float]) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    return pd.Series(data, dtype=float).dropna()


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    benchmark: float = 0.0,
    periods_per_year: int = _DAYS_PER_YEAR,
) -> Tuple[float, Dict[str, float]]:
    """Return probabilistic Sharpe ratio (PSR) and auxiliary stats."""

    series = _to_series(returns)
    n_obs = len(series)
    if n_obs < 2:
        return math.nan, {"n_obs": float(n_obs)}

    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0 or math.isnan(std):
        return math.nan, {"n_obs": float(n_obs)}

    sr = sharpe(series, periods_per_year=periods_per_year)
    if math.isnan(sr):
        return math.nan, {"n_obs": float(n_obs)}

    skew = float(((series - mean) ** 3).mean() / (std**3))
    kurt = float(((series - mean) ** 4).mean() / (std**4))
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0 or math.isnan(denom):
        return math.nan, {"n_obs": float(n_obs), "skew": skew, "kurt": kurt}

    z = (sr - benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(denom)
    return _normal_cdf(z), {
        "n_obs": float(n_obs),
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "sharpe": sr,
        "zscore": z,
    }


def permutation_sharpe_pvalue(
    returns: Iterable[float],
    periods_per_year: int = _DAYS_PER_YEAR,
    n_permutations: int = 1000,
    block_size: int = 21,
    random_state: int | None = 0,
) -> float:
    """Compute permutation p-value for Sharpe via block shuffling."""

    series = _to_series(returns)
    if series.empty:
        return math.nan

    actual_sharpe = sharpe(series, periods_per_year=periods_per_year)
    if math.isnan(actual_sharpe):
        return math.nan

    values = series.to_numpy()
    n_obs = len(values)
    block_size = int(max(1, min(block_size, n_obs)))
    rng = np.random.default_rng(random_state)
    exceed = 0
    total = 0

    blocks = [
        values[i : i + block_size] for i in range(0, n_obs, block_size)
    ]
    n_blocks = len(blocks)

    for _ in range(n_permutations):
        indices = rng.integers(0, n_blocks, size=math.ceil(n_obs / block_size))
        sampled_blocks = []
        for idx in indices:
            block = blocks[idx]
            sign = -1.0 if rng.random() < 0.5 else 1.0
            sampled_blocks.append(block * sign)
        sampled = np.concatenate(sampled_blocks)[:n_obs]
        sr = sharpe(sampled, periods_per_year=periods_per_year)
        if math.isnan(sr):
            continue
        total += 1
        if sr >= actual_sharpe:
            exceed += 1

    if total == 0:
        return math.nan
    return (exceed + 1.0) / (total + 1.0)


def run_stat_validations(
    returns: Iterable[float],
    benchmark: float = 0.0,
    periods_per_year: int = _DAYS_PER_YEAR,
    n_trials: int = 1,
    n_permutations: int = 1000,
    block_size: int = 21,
    random_state: int | None = 0,
) -> Dict[str, float]:
    """Aggregate PSR and permutation-based deflated Sharpe diagnostics."""

    series = _to_series(returns)
    result: Dict[str, float] = {
        "n_obs": float(len(series)),
    }
    psr, stats = probabilistic_sharpe_ratio(
        series,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
    )
    result.update(stats)
    result["psr"] = psr

    perm_pvalue = permutation_sharpe_pvalue(
        series,
        periods_per_year=periods_per_year,
        n_permutations=n_permutations,
        block_size=block_size,
        random_state=random_state,
    )
    result["permutation_pvalue"] = perm_pvalue

    trials = max(1, int(n_trials))
    if math.isnan(perm_pvalue):
        result["deflated_pvalue"] = math.nan
        result["deflated_confidence"] = math.nan
    else:
        adjusted = 1.0 - (1.0 - perm_pvalue) ** trials
        result["deflated_pvalue"] = adjusted
        result["deflated_confidence"] = 1.0 - adjusted

    return result
