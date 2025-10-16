from __future__ import annotations

"""Helpers to build a numeric matrix of hyperparameters for correlation analysis."""

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .run_logging import default_log_path

__all__ = [
    "load_run_history",
    "prepare_numeric_features",
    "filter_columns_by_prefix",
]


def load_run_history(log_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the persisted hyperparameter runs into a DataFrame."""

    path = Path(log_path) if log_path else default_log_path()
    if not path.exists():
        raise FileNotFoundError(f"Run history not found at {path}")
    return pd.read_csv(path)


def _object_to_numeric(series: pd.Series) -> pd.Series:
    """Best-effort conversion of object dtypes to numeric encodings."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0 and numeric.notna().sum() >= max(1, int(0.1 * len(series))):
        return numeric
    categories = series.astype("category")
    codes = categories.cat.codes.replace(-1, np.nan)
    return codes.astype(float)


def prepare_numeric_features(
    df: pd.DataFrame,
    *,
    include_metrics: bool = False,
    include_meta: bool = False,
    drop_constant: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a numeric matrix suitable for correlation analysis."""

    allowed: Iterable[str]
    if columns is not None:
        allowed = columns
    else:
        allowed = df.columns

    selected: dict[str, pd.Series] = {}
    for col in allowed:
        if not include_metrics and col.startswith("metric."):
            continue
        if not include_meta and col.startswith("meta."):
            continue
        series = df[col]
        if series.dtype == object:
            selected[col] = _object_to_numeric(series)
        elif series.dtype == bool:
            selected[col] = series.astype(int)
        else:
            selected[col] = series.astype(float) if np.issubdtype(series.dtype, np.number) else series

    matrix = pd.DataFrame(selected)
    if drop_constant and not matrix.empty:
        matrix = matrix.loc[:, matrix.apply(lambda s: s.nunique(dropna=True) > 1)]
    return matrix


def filter_columns_by_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Select columns that start with a given dotted prefix."""

    cols = [col for col in df.columns if col.startswith(prefix)]
    return df.loc[:, cols]

