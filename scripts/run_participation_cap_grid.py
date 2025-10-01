"""Run a grid over participation-cap regime highs and store artifacts."""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from dataio.config import load_config
from backtest.engine import run_backtest

PARTICIPATION_LOW = 0.025
PARTICIPATION_HIGH_VALUES: Tuple[float, ...] = (0.028, 0.030, 0.032)
ARTIFACT_JSON = Path("artifacts") / "cap_grid_v2.json"
ARTIFACT_CSV = Path("artifacts") / "cap_grid_v2.csv"


def ensure_tda_defaults(cfg: Dict) -> None:
    tda_cfg = cfg.setdefault("tda", {})
    if tda_cfg.get("smooth_span") is None:
        tda_cfg["smooth_span"] = 10
    if tda_cfg.get("eps_quantile") is None:
        tda_cfg["eps_quantile"] = 0.15


def top_entry(mapping: Dict[str, float]) -> Tuple[str | None, float | None]:
    if not mapping:
        return None, None
    key = max(mapping, key=mapping.get)
    value = float(mapping[key])
    return key, value


def run_scenario(cfg: Dict, high: float) -> Dict:
    cfg_run = deepcopy(cfg)
    ensure_tda_defaults(cfg_run)

    risk_cfg = cfg_run.setdefault("risk", {})
    participation_cfg = risk_cfg.setdefault("participation_cap_regime", {})
    participation_cfg["low"] = PARTICIPATION_LOW
    participation_cfg["high"] = float(high)

    start = time.time()
    result = run_backtest(cfg_run)
    elapsed = time.time() - start

    kpis = result.get("kpis", {})
    meta = result.get("meta", {})
    regime_controls = meta.get("regime_controls", {}) or {}
    capacity = meta.get("capacity", {}) or {}
    binding = regime_controls.get("binding") or {}

    payload = {
        "scenario": f"cap_high_{high:.3f}",
        "participation_cap_low": PARTICIPATION_LOW,
        "participation_cap_high": float(high),
        "elapsed_sec": round(elapsed, 2),
        "kpis": kpis,
        "capacity": capacity,
        "regime_controls": regime_controls,
    }

    summary = {
        "scenario": payload["scenario"],
        "sharpe": kpis.get("sharpe"),
        "max_drawdown": kpis.get("max_drawdown"),
        "cap_bind_rate": capacity.get("cap_bind_rate"),
        "asset_pre_bind_rate": capacity.get("asset_pre_bind_rate"),
        "cluster_pre_bind_rate": capacity.get("cluster_pre_bind_rate"),
    }
    print(json.dumps({"payload": summary, "binding_summary": binding.get("summary", {})}, indent=2))
    return payload


def save_artifacts(results: List[Dict]) -> None:
    artifacts_dir = ARTIFACT_JSON.parent
    artifacts_dir.mkdir(exist_ok=True)

    ARTIFACT_JSON.write_text(json.dumps(results, indent=2))

    rows = []
    for entry in results:
        kpis = entry.get("kpis", {})
        capacity = entry.get("capacity", {}) or {}
        binding = (entry.get("regime_controls", {}) or {}).get("binding") or {}

        asset_top_name, asset_top_freq = top_entry(binding.get("asset_pre_bind_frequency") or {})
        cluster_top_name, cluster_top_freq = top_entry(binding.get("cluster_pre_bind_frequency") or {})

        rows.append(
            {
                "scenario": entry["scenario"],
                "participation_cap_low": entry.get("participation_cap_low"),
                "participation_cap_high": entry.get("participation_cap_high"),
                "total_return": kpis.get("total_return"),
                "annual_return": kpis.get("annual_return"),
                "annual_vol": kpis.get("annual_vol"),
                "sharpe": kpis.get("sharpe"),
                "max_drawdown": kpis.get("max_drawdown"),
                "cap_bind_rate": capacity.get("cap_bind_rate"),
                "asset_pre_bind_rate": capacity.get("asset_pre_bind_rate"),
                "cluster_pre_bind_rate": capacity.get("cluster_pre_bind_rate"),
                "avg_cap_adjustment_l1": capacity.get("avg_cap_adjustment_l1"),
                "avg_turnover_cut_frac": capacity.get("avg_turnover_cut_frac"),
                "top_asset_pre_bind": asset_top_name,
                "top_asset_pre_bind_freq": asset_top_freq,
                "top_cluster_pre_bind": cluster_top_name,
                "top_cluster_pre_bind_freq": cluster_top_freq,
            }
        )

    pd.DataFrame(rows).to_csv(ARTIFACT_CSV, index=False)
    print(f"Wrote {ARTIFACT_JSON} and {ARTIFACT_CSV}")


def main() -> None:
    base_cfg = load_config("configs/base.yaml")
    results = [run_scenario(base_cfg, high) for high in PARTICIPATION_HIGH_VALUES]
    save_artifacts(results)


if __name__ == "__main__":
    main()
