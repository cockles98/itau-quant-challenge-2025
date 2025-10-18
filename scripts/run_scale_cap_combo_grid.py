"""Run combined grid for scale_low and participation cap high."""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from dataio.config import load_config
from backtest.engine import run_backtest

COMBOS: Tuple[Tuple[float, float], ...] = (
    (0.80, 0.032),
    (0.85, 0.032),
    (0.90, 0.032),
)
ARTIFACT_JSON = Path("artifacts") / "combo_grid_v1.json"
ARTIFACT_CSV = Path("artifacts") / "combo_grid_v1.csv"


def top_entry(mapping: Dict[str, float]) -> tuple[str | None, float | None]:
    if not mapping:
        return None, None
    key = max(mapping, key=mapping.get)
    value = float(mapping[key])
    return key, value


def run_combo(cfg: Dict, scale_low: float, cap_high: float) -> Dict:
    cfg_run = deepcopy(cfg)

    risk_cfg = cfg_run.setdefault("risk", {})
    target_vol_cfg = risk_cfg.setdefault("regime_target_vol", {})
    target_vol_cfg["scale_low"] = float(scale_low)
    target_vol_cfg.setdefault("scale_high", float(target_vol_cfg.get("scale_high", 1.2)))

    participation_cfg = risk_cfg.setdefault("participation_cap_regime", {})
    participation_cfg["low"] = float(participation_cfg.get("low", 0.025))
    participation_cfg["high"] = float(cap_high)

    start = time.time()
    result = run_backtest(cfg_run)
    elapsed = time.time() - start

    kpis = result.get("kpis", {})
    meta = result.get("meta", {})
    capacity = meta.get("capacity", {}) or {}
    regime_controls = meta.get("regime_controls", {}) or {}
    binding = regime_controls.get("binding") or {}

    payload = {
        "scenario": f"scale_{scale_low:.2f}_cap_{cap_high:.3f}",
        "scale_low": float(scale_low),
        "scale_high": float(target_vol_cfg.get("scale_high", 1.2)),
        "participation_cap_low": participation_cfg.get("low"),
        "participation_cap_high": float(cap_high),
        "elapsed_sec": round(elapsed, 2),
        "kpis": kpis,
        "capacity": capacity,
        "regime_controls": regime_controls,
    }

    summary = {
        "scenario": payload["scenario"],
        "sharpe": kpis.get("sharpe"),
        "total_return": kpis.get("total_return"),
        "max_drawdown": kpis.get("max_drawdown"),
        "cap_bind_rate": capacity.get("cap_bind_rate"),
    }
    print(json.dumps({"payload": summary, "binding": binding.get("summary", {})}, indent=2))
    return payload


def save_artifacts(results: List[Dict]) -> None:
    ARTIFACT_JSON.parent.mkdir(exist_ok=True)
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
                "scale_low": entry.get("scale_low"),
                "scale_high": entry.get("scale_high"),
                "participation_cap_low": entry.get("participation_cap_low"),
                "participation_cap_high": entry.get("participation_cap_high"),
                "total_return": kpis.get("total_return"),
                "annual_return": kpis.get("annual_return"),
                "annual_vol": kpis.get("annual_vol"),
                "sharpe": kpis.get("sharpe"),
                "max_drawdown": kpis.get("max_drawdown"),
                "cap_bind_rate": capacity.get("cap_bind_rate"),
                "asset_pre_bind_rate": capacity.get("asset_pre_bind_rate"),
                "avg_cap_adjustment_l1": capacity.get("avg_cap_adjustment_l1"),
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
    results = [run_combo(base_cfg, scale_low, cap_high) for scale_low, cap_high in COMBOS]
    save_artifacts(results)


if __name__ == "__main__":
    main()
