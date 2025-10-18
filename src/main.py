from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from backtest.engine import run_backtest
from dataio.config import load_config
from dataio.loaders import get_panel
import numpy as np
from features import compute_ph_regime_index
from reports import plot_equity_curves, table_kpis
from validation import (
    capacity_curve,
    param_sensitivity_heatmaps,
    run_purged_tuning,
    run_walk_forward,
    stress_costs,
    regime_subperiods,
)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
@contextmanager
def _temporary_argv(argv: Iterable[str]) -> Iterable[str]:
    """Temporarily patch sys.argv for script-style entry points."""
    original = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


def _invoke_script(script_name: str, args: List[str]) -> None:
    """Import a script module and invoke its main() with temporary argv."""
    module = import_module(f"scripts.{script_name}")
    script_main = getattr(module, "main", None)
    if script_main is None:
        raise RuntimeError(f"Script '{script_name}' does not expose a main() function")
    with _temporary_argv([f"{script_name}.py", *args]):
        script_main()


def run_ph_threshold_backtest_cli(config_path: str) -> None:
    logging.info("Running PH threshold backtest for configuration %s", config_path)
    _invoke_script("run_ph_threshold_backtest", ["--config", config_path])


def _synthetic_panel(start: str, end: str, periods: int = 756) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    assets = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    records = []
    for asset in assets:
        base = 100 + 5 * (ord(asset[0]) - ord("A"))
        prices = base + np.arange(len(dates)) + 0.5 * rng.standard_normal(len(dates))
        volumes = 1_000_000 + rng.integers(-50_000, 50_000, len(dates))
        for dt, price, volume in zip(dates, prices, volumes):
            records.append((dt, asset, float(price), float(abs(volume))))
    return pd.DataFrame(
        records, columns=["date", "asset", "close", "volume"]
    ).set_index(["date", "asset"])


def _load_panel(cfg: Dict) -> pd.DataFrame:
    dates_cfg = cfg.get("dates", {})
    start = dates_cfg.get("start")
    end = dates_cfg.get("end")
    if not start or not end:
        raise ValueError("Configuration must include dates.start and dates.end")
    try:
        return get_panel(start, end)
    except Exception as err:
        logging.warning("Falling back to synthetic panel: %s", err)
        panel = _synthetic_panel(start, end)
        cfg.setdefault("dates", {})
        cfg["dates"]["start"] = panel.index.get_level_values(0).min().date().isoformat()
        cfg["dates"]["end"] = panel.index.get_level_values(0).max().date().isoformat()
        return panel


def _save_series(series: pd.Series, prefix: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{prefix}.csv"
    series.to_csv(path)
    return path


def mode_backtest(cfg: Dict) -> None:
    panel = _load_panel(cfg)
    result = run_backtest(cfg, panel=panel)
    equity_path = _save_series(result["equity_curve"], "equity_curve")
    print(f"Backtest complete. Equity curve saved to {equity_path}")
    print(json.dumps(result["kpis"], indent=2))


def mode_walkforward(cfg: Dict) -> None:
    panel = _load_panel(cfg)
    wf_result = run_walk_forward(cfg, panel)
    equity_path = _save_series(wf_result["equity_curve"], "walkforward_equity")
    print(f"Walk-forward equity saved to {equity_path}")
    print(json.dumps(wf_result["kpis"], indent=2))


def mode_tune(cfg: Dict) -> None:
    panel = _load_panel(cfg)
    validation_cfg = cfg.get("validation", {})
    n_splits = int(validation_cfg.get("n_splits", 5))
    embargo = int(validation_cfg.get("embargo_days", 5))
    result = run_purged_tuning(
        cfg,
        panel,
        n_splits=n_splits,
        embargo=embargo,
        output_dir=REPORT_DIR,
    )
    print(f"Tuning complete. Results saved to {result['results_csv']}")
    print("Heatmaps:")
    for path in result["heatmaps"]:
        print("  ", path)
    print("Best parameter sets:")
    print(result["best"])
def mode_robustness(cfg: Dict) -> None:
    panel = _load_panel(cfg)
    validation_cfg = cfg.get("validation", {})
    grid = validation_cfg.get("robustness_grid")
    if not grid:
        raise ValueError("validation.robustness_grid missing from config")
    heatmap_paths = param_sensitivity_heatmaps(cfg, panel, grid)
    print("Sensitivity heatmaps:", [str(p) for p in heatmap_paths])
    stress_path = stress_costs(
        cfg, panel, validation_cfg.get("stress_multipliers", [0.5, 1.0, 2.0])
    )
    print("Stress cost table:", stress_path)
    # Regime subperiods using PH turbulence regime index
    prices = panel["close"].unstack("asset").sort_index()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regime_series = compute_ph_regime_index(returns, cfg)
    regime_table = regime_subperiods(cfg, panel, regime_series)
    regime_path = REPORT_DIR / "regime_kpis.csv"
    regime_table.to_csv(regime_path)
    print("Regime KPI table:", regime_path)


def mode_capacity(cfg: Dict) -> None:
    panel = _load_panel(cfg)
    caps = cfg.get("validation", {}).get("participation_caps", [0.01, 0.02, 0.05, 0.10])
    outputs = capacity_curve(cfg, panel, participation_caps=caps)
    print("Capacity outputs:", {k: str(v) for k, v in outputs.items()})


def mode_report(cfg: Dict) -> None:
    panel = _load_panel(cfg)
    result = run_backtest(cfg, panel=panel)
    equity = result["equity_curve"]
    returns = equity.pct_change().dropna()
    curves = {"Strategy": equity}
    plot_path = plot_equity_curves(curves, filename="equity_curves_full.png")
    table = table_kpis({"Strategy": {"equity_curve": equity, "returns": returns}})
    print("Report generated:", plot_path)
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas pipeline entrypoint")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "backtest",
            "walkforward",
            "tune",
            "robustness",
            "capacity",
            "report",
        ],
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument(
        "--ph-threshold-bt",
        action="store_true",
        help="Backtest PH turbulence thresholds as a regime filter.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = load_config(args.config)
    config_path = str(args.config)

    if args.mode == "backtest":
        mode_backtest(cfg)
    elif args.mode == "walkforward":
        mode_walkforward(cfg)
    elif args.mode == "tune":
        mode_tune(cfg)
    elif args.mode == "robustness":
        mode_robustness(cfg)
    elif args.mode == "capacity":
        mode_capacity(cfg)
    elif args.mode == "report":
        mode_report(cfg)

    if args.ph_threshold_bt:
        run_ph_threshold_backtest_cli(config_path)


if __name__ == "__main__":
    main()
