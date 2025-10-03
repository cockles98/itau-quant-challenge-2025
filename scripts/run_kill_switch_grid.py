#!/usr/bin/env python
"""Grid search for kill-switch thresholds using walk-forward OOS metrics."""
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from dataio.config import load_config
from dataio.loaders import get_panel
from metrics import avg_time_under_water, max_time_under_water
from validation.walk_forward import run_walk_forward

DEFAULT_LOOKBACK: Tuple[int, ...] = (60, 90, 120)
DEFAULT_MDD_THRES: Tuple[float, ...] = (-0.15, -0.20, -0.25)
DEFAULT_VOL_MULT: Tuple[float, ...] = (1.5, 1.8, 2.0)
DEFAULT_JSON = Path("artifacts") / "kill_switch_grid.json"
DEFAULT_CSV = Path("artifacts") / "kill_switch_grid.csv"


def _maybe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else math.nan
    return math.nan


def _finite(value: float | None, fallback: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return fallback


def _config_dates(cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dates = cfg.get("dates", {})
    start = pd.Timestamp(dates.get("start"))
    end = pd.Timestamp(dates.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Config must include dates.start and dates.end")
    return start, end


def _run_combo(
    cfg: Dict,
    panel: pd.DataFrame,
    lookback: int,
    thres: float,
    vol_mult: float,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    max_windows: int | None = None,
) -> Dict[str, float]:
    cfg_run = deepcopy(cfg)
    if date_start or date_end:
        dates_cfg = cfg_run.setdefault("dates", {})
        if date_start:
            dates_cfg["start"] = date_start
        if date_end:
            dates_cfg["end"] = date_end
    mw = max_windows if max_windows and max_windows > 0 else None
    panel_slice = panel
    if date_start or date_end:
        level0 = panel.index.get_level_values(0)
        start_ts = pd.Timestamp(date_start) if date_start else level0.min()
        end_ts = pd.Timestamp(date_end) if date_end else level0.max()
        mask = (level0 >= start_ts) & (level0 <= end_ts)
        panel_slice = panel.loc[mask]
    if panel_slice.empty:
        raise ValueError("Panel slice is empty for selected date range")
    risk_cfg = cfg_run.setdefault("risk", {})
    risk_cfg["mdd_lookback"] = int(lookback)
    risk_cfg["mdd_thres"] = float(thres)
    risk_cfg["vol_mult"] = float(vol_mult)

    result = run_walk_forward(cfg_run, panel_slice, max_windows=mw)
    equity = result.get("equity_curve")
    if equity is None or equity.empty:
        raise ValueError("Walk-forward returned empty equity curve")

    avg_tuw = avg_time_under_water(equity)
    max_tuw = max_time_under_water(equity)
    kpis = result.get("kpis", {})

    return {
        "scenario": f"lb_{lookback}_thr_{thres:.2f}_vol_{vol_mult:.2f}",
        "mdd_lookback": int(lookback),
        "mdd_thres": float(thres),
        "vol_mult": float(vol_mult),
        "avg_time_under_water": float(avg_tuw),
        "max_time_under_water": float(max_tuw),
        "sharpe": _maybe_float(kpis.get("Sharpe")),
        "max_drawdown": _maybe_float(kpis.get("MaxDD")),
    }




def _passes_filters(entry: Dict[str, float], min_sharpe: float, max_drawdown: float | None) -> bool:
    sharpe = entry.get("sharpe")
    if math.isfinite(min_sharpe):
        if math.isnan(sharpe) or sharpe < min_sharpe:
            return False
    if max_drawdown is not None and math.isfinite(max_drawdown):
        md = entry.get("max_drawdown")
        if math.isnan(md) or abs(md) > max_drawdown:
            return False
    return True


def _sort_key(entry: Dict[str, float]) -> Tuple[float, float, float, float]:
    avg_tuw = _finite(entry.get("avg_time_under_water"), math.inf)
    sharpe = -_finite(entry.get("sharpe"), -math.inf)
    max_tuw = _finite(entry.get("max_time_under_water"), math.inf)
    max_dd = _finite(abs(entry.get("max_drawdown", math.nan)), math.inf)
    return avg_tuw, sharpe, max_tuw, max_dd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search kill-switch thresholds using walk-forward validation."
    )
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to config YAML (default: %(default)s)"
    )
    parser.add_argument(
        "--mdd-lookback",
        type=int,
        nargs="+",
        default=DEFAULT_LOOKBACK,
        metavar="N",
        help="Lookback lengths to test (default: %(default)s)",
    )
    parser.add_argument(
        "--mdd-thres",
        type=float,
        nargs="+",
        default=DEFAULT_MDD_THRES,
        metavar="X",
        help="Drawdown thresholds to test (default: %(default)s)",
    )
    parser.add_argument(
        "--vol-mult",
        type=float,
        nargs="+",
        default=DEFAULT_VOL_MULT,
        metavar="X",
        help="Volatility multiples to test (default: %(default)s)",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum Sharpe required for eligibility (default: %(default)s)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        help="Maximum allowed drawdown magnitude (e.g. 0.15 = 15%%). Use a negative value to disable.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="Limit number of parameter combinations evaluated (0 = all).",
    )
    parser.add_argument(
        "--stop-on-first",
        action="store_true",
        help="Stop the search once a scenario passes the Sharpe/drawdown filters.",
    )
    parser.add_argument(
        "--date-start",
        type=str,
        default=None,
        help="Override data start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="Override data end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Limit number of walk-forward windows evaluated (0 = all).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path for JSON artefact (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path for CSV artefact (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.date_start or args.date_end:
        dates_cfg = cfg.setdefault("dates", {})
        if args.date_start:
            dates_cfg["start"] = args.date_start
        if args.date_end:
            dates_cfg["end"] = args.date_end
    start, end = _config_dates(cfg)
    panel = get_panel(start, end)

    date_start = args.date_start
    date_end = args.date_end
    max_windows = args.max_windows if args.max_windows and args.max_windows > 0 else None

    combos = list(product(args.mdd_lookback, args.mdd_thres, args.vol_mult))
    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]

    dd_limit = None if args.max_drawdown is not None and args.max_drawdown < 0 else args.max_drawdown

    results: List[Dict[str, float]] = []
    for lookback, thres, vol_mult in combos:
        start_ts = time.time()
        entry = _run_combo(
            cfg,
            panel,
            lookback,
            thres,
            vol_mult,
            date_start=date_start,
            date_end=date_end,
            max_windows=max_windows,
        )
        entry["elapsed_sec"] = round(time.time() - start_ts, 2)
        print(json.dumps({
            "scenario": entry["scenario"],
            "avg_time_under_water": entry["avg_time_under_water"],
            "max_time_under_water": entry["max_time_under_water"],
            "sharpe": entry.get("sharpe"),
            "max_drawdown": entry.get("max_drawdown"),
        }, indent=2))
        results.append(entry)
        if args.stop_on_first and _passes_filters(entry, args.min_sharpe, dd_limit):
            print("Stopping early because scenario satisfies filters.")
            break

    if not results:
        raise RuntimeError("No scenarios were evaluated")

    ordered = sorted(results, key=_sort_key)
    eligible = [entry for entry in ordered if _passes_filters(entry, args.min_sharpe, dd_limit)]
    if eligible:
        best = eligible[0]
    else:
        print("No scenario passed eligibility filters; falling back to best overall by regret order.")
        best = ordered[0]
    print("Best scenario:")
    print(json.dumps(best, indent=2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
