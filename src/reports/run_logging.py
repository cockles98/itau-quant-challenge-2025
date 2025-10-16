from __future__ import annotations

"""Utilities to persist hyperparameter configurations and KPIs for later analysis."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableSequence, Sequence

import pandas as pd

__all__ = ["flatten_config", "append_run_log", "default_log_path"]

_REPORTS_ROOT = Path(__file__).resolve().parents[2] / "reports"


def default_log_path() -> Path:
    """Return the canonical CSV path that stores the hyperparameter run history."""

    return _REPORTS_ROOT / "hyperparameter_runs.csv"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def flatten_config(
    mapping: Mapping[str, Any],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten nested dictionaries/lists into a dotted-key mapping.

    Lists and tuples are expanded with integer suffixes (e.g. ``alpha.0``).
    """

    items: Dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_config(value, parent_key=new_key, sep=sep))
        elif _is_sequence(value):
            for idx, element in enumerate(value):
                items.update(
                    flatten_config(
                        {str(idx): element},
                        parent_key=new_key,
                        sep=sep,
                    )
                )
        else:
            items[new_key] = value
    return items


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure values are serialisable for CSV storage."""

    normalised: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float)) or value is None:
            normalised[key] = value
        elif isinstance(value, bool):
            normalised[key] = int(value)
        elif isinstance(value, (datetime,)):
            normalised[key] = value.isoformat()
        else:
            normalised[key] = repr(value)
    return normalised


def append_run_log(
    config: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    log_path: Path | str | None = None,
) -> Path:
    """Append a run entry with the flattened configuration and metrics to the CSV log."""

    target = Path(log_path) if log_path else default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {}
    payload.update(flatten_config(config))

    if metrics:
        payload.update({f"metric.{k}": v for k, v in metrics.items()})

    meta_dict: Dict[str, Any] = dict(meta or {})
    meta_dict.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    payload.update({f"meta.{k}": v for k, v in meta_dict.items()})

    normalised = _normalise_row(payload)
    header = not target.exists()
    df = pd.DataFrame([normalised])
    df.to_csv(target, mode="a", header=header, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return target

