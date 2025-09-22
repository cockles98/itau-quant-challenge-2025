from __future__ import annotations

import numbers
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config"]


class ConfigError(ValueError):
    """Raised when the configuration file fails validation."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate required keys."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file '{config_path}' does not exist.")

    raw = config_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as err:
        raise ConfigError(f"Failed to parse YAML: {err}") from err

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping of keys to values.")

    _validate_config(data)
    return data


def _validate_config(cfg: dict[str, Any]) -> None:
    seeds = _require_type(cfg, "seeds", int)
    if seeds < 0:
        raise ConfigError("seeds must be a non-negative integer.")

    dates = _require_mapping(cfg, "dates")
    _validate_date(dates, "start")
    _validate_date(dates, "end")

    tda = _require_mapping(cfg, "tda")
    _require_positive_int(tda, "delay")
    _require_positive_int(tda, "dim")
    _require_positive_int(tda, "n_cubes")
    _require_non_negative_real(tda, "overlap")

    factors = _require_mapping(cfg, "factors")
    alphas = _require_type(factors, "alphas", list)
    if not alphas:
        raise ConfigError("factors.alphas must contain at least one entry.")
    if not all(isinstance(alpha, str) for alpha in alphas):
        raise ConfigError("factors.alphas must be a list of strings.")

    windows = _require_mapping(cfg, "windows")
    _require_positive_int(windows, "vol_window")
    _require_positive_int(windows, "atr_len")

    costs = _require_mapping(cfg, "costs")
    _require_non_negative_real(costs, "fee_bps")

    _require_non_negative_real(cfg, "participation_cap")
    _require_non_negative_real(cfg, "turnover_cap")
    _require_non_negative_real(cfg, "vol_target")


def _require_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    return _require_type(container, key, dict)


def _require_positive_int(container: dict[str, Any], key: str) -> int:
    value = _require_type(container, key, int)
    if value <= 0:
        raise ConfigError(f"{key} must be a positive integer.")
    return value


def _require_non_negative_real(container: dict[str, Any], key: str) -> float:
    value = _require_number(container, key)
    if value < 0:
        raise ConfigError(f"{key} must be non-negative.")
    return float(value)


def _require_number(container: dict[str, Any], key: str) -> numbers.Real:
    if key not in container:
        raise ConfigError(f"Missing required configuration key '{key}'.")
    value = container[key]
    if not isinstance(value, numbers.Real):
        raise ConfigError(f"{key} must be a numeric value.")
    return value


def _require_type(container: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in container:
        raise ConfigError(f"Missing required configuration key '{key}'.")
    value = container[key]
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"{key} must be of type {expected_type.__name__}, got {type(value).__name__}."
        )
    return value


def _validate_date(container: dict[str, Any], key: str) -> None:
    if key not in container:
        raise ConfigError(f"Missing required configuration key '{key}'.")
    value = container[key]
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except ValueError as err:
            raise ConfigError(f"{key} must be an ISO date string (YYYY-MM-DD).") from err
    elif isinstance(value, (datetime, date)):
        return
    else:
        raise ConfigError(
            f"{key} must be an ISO date string (YYYY-MM-DD) or a date/datetime object, got {type(value).__name__}."
        )
