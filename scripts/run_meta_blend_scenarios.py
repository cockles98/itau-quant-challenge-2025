"""Run multiple meta-blend configurations and summarise KPIs."""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from backtest.engine import run_backtest
from dataio.config import load_config
from dataio.loaders import get_panel

DEFAULT_SCENARIOS: List[Dict[str, object]] = [
    {
        "name": "ridge_h21",
        "model_type": "ridge",
        "horizon": 21,
        "alpha_grid": [0.02, 0.05, 0.1, 0.2],
        "rolling_window": 504,
    },
    {
        "name": "ridge_h10",
        "model_type": "ridge",
        "horizon": 10,
        "alpha_grid": [0.02, 0.05, 0.1, 0.2],
        "rolling_window": 378,
    },
    {
        "name": "elastic_h21",
        "model_type": "elasticnet",
        "horizon": 21,
        "alpha_grid": [0.02, 0.05, 0.1],
        "l1_ratio_grid": [0.2, 0.5, 0.8],
        "rolling_window": 504,
    },
]


def _apply_scenario(base_cfg: Dict[str, object], scenario: Dict[str, object]) -> Dict[str, object]:
    cfg = deepcopy(base_cfg)
    factors_cfg = cfg.setdefault("factors", {})
    meta_cfg = factors_cfg.setdefault("meta_blend", {})
    meta_cfg.update(
        {
            "enabled": True,
            "save_artifacts": True,
            "artifact_prefix": scenario["name"],
            "model_type": scenario["model_type"],
            "horizon": scenario["horizon"],
            "alpha_grid": scenario.get("alpha_grid", meta_cfg.get("alpha_grid")),
            "l1_ratio_grid": scenario.get("l1_ratio_grid", meta_cfg.get("l1_ratio_grid")),
            "rolling_window": scenario.get("rolling_window", meta_cfg.get("rolling_window")),
        }
    )
    return cfg


def _ensure_iterable(obj: Iterable[Dict[str, object]] | None) -> List[Dict[str, object]]:
    if obj is None:
        return []
    return list(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple meta-blend scenarios")
    parser.add_argument("--config", default="configs/base.yaml", help="Base YAML config")
    parser.add_argument(
        "--outdir",
        default="reports",
        help="Directory to store scenario summaries",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    dates_cfg = base_cfg.get("dates", {})
    start = dates_cfg.get("start")
    end = dates_cfg.get("end")
    if not start or not end:
        raise ValueError("Configuration must include dates.start and dates.end")

    panel = get_panel(start, end)

    summaries = []
    meta_paths: Dict[str, str] = {}

    scenarios = DEFAULT_SCENARIOS

    for scenario in scenarios:
        cfg = _apply_scenario(base_cfg, scenario)
        scenario_name = scenario["name"]
        print(f"Running scenario {scenario_name}...")
        start_time = time.time()
        result = run_backtest(cfg, panel=panel)
        elapsed = time.time() - start_time
        kpis = result.get("kpis", {}) or {}
        meta = result.get("meta", {}) or {}
        meta_blend = meta.get("meta_blend", {}) or {}
        if meta_blend.get("artifact_path"):
            meta_paths[scenario_name] = meta_blend["artifact_path"]
        summaries.append(
            {
                "scenario": scenario_name,
                "elapsed_sec": round(elapsed, 2),
                **{key: kpis.get(key) for key in kpis},
            }
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    table_path = outdir / f"meta_blend_scenarios_{timestamp}.csv"
    pd.DataFrame(summaries).to_csv(table_path, index=False)

    meta_index_path = outdir / f"meta_blend_artifacts_{timestamp}.json"
    with meta_index_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_paths, handle, indent=2)

    print(f"Scenario summary saved to {table_path}")
    if meta_paths:
        print(f"Meta-blend artifacts indexed at {meta_index_path}")


if __name__ == "__main__":
    main()
