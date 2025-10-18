from __future__ import annotations

"""Lightweight end-to-end smoke test for the Atlas pipeline.

This script generates a synthetic multi-asset panel (~3 years of business days),
runs a short backtest with two or more monthly rebalances, and prints a compact
summary featuring:
  * Last five observations of the PH turbulence z-score series
  * Headline Mapper metrics (n_components, avg_degree, gini_node_size)
  * Final HRP topological allocation
  * Core KPIs from the backtest engine

The goal is to validate that the full stack (data preparation, TDA features,
HRP allocation, risk controls) executes successfully without proprietary data.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from dataio.config import load_config
from features.regime.ph_regime import compute_ph_regime_index

matplotlib.use("Agg", force=True)

def _synthetic_panel(
    start: str,
    periods: int,
    tickers: Iterable[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic OHLCV panel with log-normal price paths."""
    dates = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(seed)
    records: List[tuple[pd.Timestamp, str, float, float]] = []

    for ticker in tickers:
        daily_ret = rng.normal(loc=0.0006, scale=0.012, size=len(dates))
        prices = 100.0 * np.exp(np.cumsum(daily_ret))
        volumes = rng.integers(low=400_000, high=900_000, size=len(dates))
        for dt, price, volume in zip(dates, prices, volumes):
            records.append((dt, ticker, float(price), float(volume)))

    panel = pd.DataFrame(records, columns=["date", "asset", "close", "volume"])
    return panel.set_index(["date", "asset"]).sort_index()


def _prepare_config(base_cfg: Dict, start: pd.Timestamp, end: pd.Timestamp, n_assets: int) -> Dict:
    """Adjust base configuration for synthetic data windows."""
    cfg = deepcopy(base_cfg)
    cfg.setdefault("dates", {})
    cfg["dates"]["start"] = start.date().isoformat()
    cfg["dates"]["end"] = end.date().isoformat()

    universe_cfg = cfg.setdefault("universe", {})
    universe_cfg["top_n"] = n_assets
    universe_cfg["adv_min"] = 0.0
    universe_cfg["price_min"] = 0.0
    universe_cfg["age_min"] = 1
    universe_cfg["hysteresis_rebalances"] = 1

    mapper_cfg = cfg.setdefault("mapper", {})
    mapper_cfg.setdefault("lookback", 126)
    mapper_cfg.setdefault("min_history", 90)

    risk_cfg = cfg.setdefault("risk", {})
    risk_cfg.setdefault("target_vol", 0.10)
    risk_cfg.setdefault("turnover_cap", 0.25)
    risk_cfg.setdefault("participation_cap", 0.05)
    risk_cfg.setdefault("regime_scale_low", 0.8)
    risk_cfg.setdefault("regime_scale_high", 1.2)

    return cfg


def main() -> None:
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    periods = 2 * 252  # ~2 anos úteis (suficiente para alguns rebalanceamentos)
    panel = _synthetic_panel(start="2020-01-01", periods=periods, tickers=tickers)

    config_path = Path("configs/base.yaml")
    base_cfg = load_config(str(config_path))
    cfg = _prepare_config(
        base_cfg,
        start=panel.index.get_level_values(0).min(),
        end=panel.index.get_level_values(0).max(),
        n_assets=len(tickers),
    )

    print("[SMOKE] Running lightweight backtest...")
    result = run_backtest(cfg, panel=panel)

    prices = panel["close"].unstack("asset").sort_index()
    returns = prices.pct_change().iloc[1:].dropna(how="any")
    ph_series = compute_ph_regime_index(returns, cfg).ffill().rename("PH_turbulence_z")
    print("\n[SMOKE] Últimas 5 observações de PH_turbulence_z:")
    print(ph_series.tail())

    mapper_meta = result.get("meta", {}).get("tda_topology", {}) or {}
    mapper_metrics_path = mapper_meta.get("mapper_metrics_path")
    if mapper_metrics_path and Path(mapper_metrics_path).exists():
        mapper_df = pd.read_csv(mapper_metrics_path)
        mapper_tail = mapper_df.tail().set_index("date")
        print("\n[SMOKE] Métricas do Mapper (últimas datas):")
        print(mapper_tail[["n_components", "avg_degree", "gini_node_size"]])
    else:
        print("\n[SMOKE] Métricas do Mapper indisponíveis.")

    weights = result.get("weights")
    if isinstance(weights, pd.Series):
        print("\n[SMOKE] Alocação HRP-topo final:")
        print(weights.sort_values(ascending=False))

    kpis = result.get("kpis", {})
    print("\n[SMOKE] KPIs básicos:")
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
