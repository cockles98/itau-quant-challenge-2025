#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

CHECKS = [
    {
        "name": "ph_turbulence.py updated",
        "paths": ["src/features/tda/ph_turbulence.py"],
    },
    {
        "name": "mapper.py updated",
        "paths": ["src/features/tda/mapper.py"],
    },
    {
        "name": "peripherality.py present",
        "paths": ["src/features/factors/peripherality.py"],
    },
    {
        "name": "engine integrates new modules",
        "paths": ["src/backtest/engine.py"],
        "contains": ["apply_periphery_bias", "guards.rolling_mdd_kill_switch", "regime_controls_path"],
    },
    {
        "name": "artifacts/tda directory",
        "paths": ["artifacts/tda"],
        "is_dir": True,
    },
    {
        "name": "regime_kpis.csv emitted",
        "paths": ["reports"],
        "glob": "regime_kpis*.csv",
    },
    {
        "name": "tda_dashboard.png emitted",
        "paths": ["reports"],
        "glob": "tda_dashboard*.png",
    },
]


def check_exists(path: Path, is_dir: bool = False) -> bool:
    if is_dir:
        return path.exists() and path.is_dir()
    return path.exists()


def check_contains(path: Path, substrings: List[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return all(sub in text for sub in substrings)


def check_glob(path: Path, pattern: str) -> bool:
    return any(path.glob(pattern))


def run_check(check: Dict) -> bool:
    paths = [Path(p) for p in check.get("paths", [])]
    is_dir = check.get("is_dir", False)
    contains = check.get("contains")
    glob_pattern = check.get("glob")

    if glob_pattern:
        base = paths[0] if paths else Path(".")
        return check_glob(base, glob_pattern)

    if contains:
        for path in paths:
            if not check_contains(path, contains):
                return False
        return True

    for path in paths:
        if not check_exists(path, is_dir=is_dir):
            return False
    return True


def main() -> int:
    results = []
    for spec in CHECKS:
        ok = run_check(spec)
        results.append((spec["name"], ok))

    overall_ok = True
    for name, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            overall_ok = False

    metadata = {name: ok for name, ok in results}
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "verify_tda_upgrade.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
