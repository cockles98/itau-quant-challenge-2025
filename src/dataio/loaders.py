from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

__all__ = ["get_panel", "get_adv", "select_universe"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
_CACHE_DIR = _ARTIFACTS_DIR / "cache"


def get_panel(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Load OHLCV panel data between *start* and *end* dates.

    The loader expects local CSV files under ``/data`` containing the columns
    ``date`` (or ``timestamp``), ``close`` and ``volume``. An ``asset`` column is
    optional-when missing, the CSV stem is used as the asset identifier.

    The result is cached as Parquet under ``/artifacts/cache`` so repeated calls
    with the same date range re-use the cached payload until a newer CSV is
    detected.
    """

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError("start must not be after end")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = _panel_cache_path(start_ts, end_ts)
    source_files = _get_source_files()
    if cache_path.exists() and _is_cache_valid(cache_path, source_files):
        try:
            cached = pd.read_parquet(cache_path)
            return _format_panel(cached)
        except (ImportError, ValueError, OSError):
            pass  # Fallback to reloading from CSVs

    frames: List[pd.DataFrame] = []
    for csv_path in source_files:
        frame = pd.read_csv(csv_path)
        frame = frame.rename(columns={"timestamp": "date"})
        if "date" not in frame.columns:
            raise ValueError(f"Missing 'date' column in {csv_path}")
        if "asset" not in frame.columns:
            frame["asset"] = csv_path.stem
        required_cols = {"close", "volume"}
        missing = required_cols - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} in {csv_path}")
        frame["date"] = pd.to_datetime(frame["date"], utc=False).dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frames.append(frame[["date", "asset", "close", "volume"]])

    if not frames:
        raise FileNotFoundError("No CSV files found under the local data directory")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.loc[(panel["date"] >= start_ts) & (panel["date"] <= end_ts)]
    if panel.empty:
        raise ValueError("No data available in the requested date range")

    panel = _format_panel(panel)

    try:
        panel.reset_index().to_parquet(cache_path, index=False)
    except (ImportError, ValueError, OSError):
        # Parquet engines are optional; if unavailable we silently skip caching.
        pass

    return panel


def get_adv(panel: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Compute the average daily volume (ADV) per asset.

    Returns a Pandas Series with the same MultiIndex as *panel*. The rolling
    window ignores periods before enough history is available, yielding NaNs.
    """

    if lookback <= 0:
        raise ValueError("lookback must be a positive integer")
    _ensure_panel(panel)

    volume_wide = panel.sort_index().loc[:, "volume"].unstack("asset")
    adv_wide = volume_wide.rolling(window=lookback, min_periods=lookback).mean()
    adv = adv_wide.stack().rename("adv")
    adv.index.set_names(["date", "asset"], inplace=True)
    return adv


def select_universe(
    panel: pd.DataFrame,
    *,
    top_n: int = 20,
    adv_min: float,
    price_min: float,
    age_min: int,
    hysteresis_rebalances: int = 2,
    calendar: str = "B",
) -> Dict[pd.Timestamp, List[str]]:
    """Select the trading universe with hysteresis on entries and exits.

    Parameters
    ----------
    panel : pd.DataFrame
        MultiIndex DataFrame (date, asset) with ``close`` and ``volume`` columns.
    top_n : int, default 20
        Maximum number of assets admitted per rebalance based on ADV ranking.
    adv_min : float
        Minimum ADV required for an asset to be considered eligible.
    price_min : float
        Minimum close price required for eligibility.
    age_min : int
        Minimum number of observations since inception for the asset.
    hysteresis_rebalances : int, default 2
        Number of consecutive rebalances needed for an asset to enter or exit.
    calendar : str, default "B"
        Pandas offset string governing the rebalance schedule.

    Returns
    -------
    dict
        Mapping from scheduled rebalance dates to the ordered list of active
        tickers after the hysteresis logic is applied.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if age_min <= 0:
        raise ValueError("age_min must be positive")
    if adv_min < 0:
        raise ValueError("adv_min must be non-negative")
    if price_min < 0:
        raise ValueError("price_min must be non-negative")
    if hysteresis_rebalances <= 0:
        raise ValueError("hysteresis_rebalances must be positive")

    _ensure_panel(panel)

    working = panel.sort_index().copy()
    working["adv"] = get_adv(working)
    working["age"] = working.groupby(level="asset").cumcount() + 1

    unique_dates = working.index.get_level_values("date").unique().sort_values()
    if unique_dates.empty:
        return {}

    schedule = pd.date_range(unique_dates.min(), unique_dates.max(), freq=calendar)
    if len(schedule) == 0:
        return {}

    results: Dict[pd.Timestamp, List[str]] = {}
    active_assets: set[str] = set()
    enter_streak: defaultdict[str, int] = defaultdict(int)
    exit_streak: defaultdict[str, int] = defaultdict(int)

    for rebalance_date in schedule:
        trade_candidates = unique_dates[unique_dates <= rebalance_date]
        if trade_candidates.empty:
            continue
        trade_date = trade_candidates[-1]
        snapshot = working.xs(trade_date)
        snapshot = snapshot.dropna(subset=["close", "volume", "adv"])

        eligible = snapshot[
            (snapshot["close"] >= price_min)
            & (snapshot["adv"] >= adv_min)
            & (snapshot["age"] >= age_min)
        ]
        eligible = eligible.sort_values("adv", ascending=False)
        candidate_assets = eligible.index[:top_n]
        candidate_set = set(candidate_assets)

        for asset in candidate_set:
            enter_streak[asset] += 1
            exit_streak[asset] = 0

        considered_assets = set(enter_streak.keys()) | active_assets
        for asset in considered_assets - candidate_set:
            enter_streak[asset] = 0
            if asset in active_assets:
                exit_streak[asset] += 1
            else:
                exit_streak[asset] = 0

        for asset in candidate_set:
            if enter_streak[asset] >= hysteresis_rebalances:
                active_assets.add(asset)

        for asset in list(active_assets):
            if (
                asset not in candidate_set
                and exit_streak[asset] >= hysteresis_rebalances
            ):
                active_assets.remove(asset)
                exit_streak[asset] = 0

        if not active_assets:
            results[rebalance_date] = []
            continue

        ordered = []
        if not snapshot.empty:
            snapshot_active = snapshot.loc[snapshot.index.intersection(active_assets)]
            snapshot_active = snapshot_active.sort_values("adv", ascending=False)
            ordered.extend(snapshot_active.index.tolist())

        missing = [asset for asset in sorted(active_assets) if asset not in ordered]
        ordered.extend(missing)

        results[rebalance_date] = ordered

    return results


def _ensure_panel(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("panel must have a MultiIndex on (date, asset)")
    if frame.index.names != ["date", "asset"]:
        raise ValueError("panel index must be named 'date' and 'asset'")
    missing_columns = {"close", "volume"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"panel is missing required columns: {missing_columns}")


def _format_panel(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    if "date" not in formatted.columns or "asset" not in formatted.columns:
        raise ValueError("Expected 'date' and 'asset' columns for panel formatting")
    formatted["date"] = pd.to_datetime(formatted["date"], utc=False).dt.normalize()
    formatted = formatted.sort_values(["date", "asset"]).set_index(["date", "asset"])
    formatted.index.set_names(["date", "asset"], inplace=True)
    return formatted[["close", "volume"]]


def _get_source_files() -> List[Path]:
    if not _DATA_DIR.exists():
        raise FileNotFoundError(
            "Data directory '/data' not found. Place CSV files locally before loading."
        )
    files = sorted(_DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found under the local data directory")
    return files


def _panel_cache_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    name = f"panel_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    return _CACHE_DIR / name


def _is_cache_valid(cache_path: Path, sources: Iterable[Path]) -> bool:
    try:
        cache_mtime = cache_path.stat().st_mtime
    except FileNotFoundError:
        return False
    source_mtime = max(path.stat().st_mtime for path in sources)
    return cache_mtime >= source_mtime
