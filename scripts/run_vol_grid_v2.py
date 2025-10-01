"""Run a grid over regime target vol scale_low values and store artifacts."""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Dict

import pandas as pd

from dataio.config import load_config
from backtest.engine import run_backtest

SCALE_LOW_VALUES: tuple[float, ...] = (0.80, 0.85, 0.90)
SCALE_HIGH = 1.20
ARTIFACT_JSON = Path("artifacts") / "vol_grid_v2.json"
ARTIFACT_CSV = Path("artifacts") / "vol_grid_v2.csv"

def ensure_tda_defaults(cfg: Dict) -> None:
    tda_cfg = cfg.setdefault("tda", {})
    if tda_cfg.get("smooth_span") is None:
        tda_cfg["smooth_span"] = 10
    if tda_cfg.get("eps_quantile") is None:
        tda_cfg["eps_quantile"] = 0.15

def run_scenario(cfg: Dict, scale_low: float) -> Dict:
    cfg_run = deepcopy(cfg)
    ensure_tda_defaults(cfg_run)
    regime_cfg = cfg_run.setdefault("risk", {}).setdefault("regime_target_vol", {})
    regime_cfg["scale_low"] = scale_low
    regime_cfg["scale_high"] = SCALE_HIGH

    start = time.time()
    result = run_backtest(cfg_run)
    elapsed = time.time() - start

    kpis = result.get("kpis", {})
    meta = result.get("meta", {})
    regime_controls = meta.get("regime_controls", {})

    payload = {
        "scenario": f"scale_low_{scale_low:.2f}",
        "scale_low": scale_low,
        "scale_high": SCALE_HIGH,
        "smooth_span": cfg_run["tda"].get("smooth_span"),
        "eps_quantile": cfg_run["tda"].get("eps_quantile"),
        "elapsed_sec": round(elapsed, 2),
        "kpis": kpis,
        "regime_controls": regime_controls,
    }
    print(json.dumps(payload, indent=2))
    return payload

def save_artifacts(results: List[Dict]) -> None:
    artifacts_dir = ARTIFACT_JSON.parent
    artifacts_dir.mkdir(exist_ok=True)

    ARTIFACT_JSON.write_text(json.dumps(results, indent=2))

    rows = []
    for entry in results:
        kpis = entry.get("kpis", {})
        regime_controls = entry.get("regime_controls") or {}
        rows.append(
            {
                "scenario": entry["scenario"],
                "scale_low": entry["scale_low"],
                "scale_high": entry["scale_high"],
                "smooth_span": entry["smooth_span"],
                "eps_quantile": entry["eps_quantile"],
                "total_return": kpis.get("total_return"),
                "annual_return": kpis.get("annual_return"),
                "annual_vol": kpis.get("annual_vol"),
                "sharpe": kpis.get("sharpe"),
                "max_drawdown": kpis.get("max_drawdown"),
                "cap_bind_rate": regime_controls.get("cap_bind_rate"),
                "asset_pre_bind_mean": regime_controls.get("asset_pre_bind_mean"),
                "cluster_pre_bind_mean": regime_controls.get("cluster_pre_bind_mean"),
                "participation_cap_low": regime_controls.get("participation_cap_low"),
                "participation_cap_high": regime_controls.get("participation_cap_high"),
            }
        )
    pd.DataFrame(rows).to_csv(ARTIFACT_CSV, index=False)

    print(f"Wrote {ARTIFACT_JSON} and {ARTIFACT_CSV}")

def main() -> None:
    base_cfg = load_config("configs/base.yaml")
    results = [run_scenario(base_cfg, value) for value in SCALE_LOW_VALUES]
    save_artifacts(results)

if __name__ == "__main__":
    main()
