import json
import time
from copy import deepcopy

from dataio.config import load_config
from backtest.engine import run_backtest

BASE_TDA = {
    "delay": 1,
    "dim": 2,
    "n_cubes": 6,
    "overlap": 0.5,
    "epsilon": None,
    "min_samples": 4,
    "window": 252,
}

SCENARIOS = [
    ("baseline", {}),
    ("n_cubes=2", {"n_cubes": 2}),
    ("n_cubes=4", {"n_cubes": 4}),
    ("n_cubes=8", {"n_cubes": 8}),
    ("n_cubes=12", {"n_cubes": 12}),
    ("overlap=0.25", {"overlap": 0.25}),
    ("overlap=0.75", {"overlap": 0.75}),
    ("overlap=1.0", {"overlap": 1.0}),
    ("epsilon=0.25", {"epsilon": 0.25}),
    ("epsilon=0.5", {"epsilon": 0.5}),
    ("epsilon=0.75", {"epsilon": 0.75}),
]

results = []

for name, overrides in SCENARIOS:
    cfg = load_config('configs/base.yaml')
    cfg['dates']['start'] = '2021-01-04'
    cfg['dates']['end'] = '2022-12-02'
    cfg.setdefault('tda', {}).update(BASE_TDA)
    cfg['tda'].update(overrides)
    start = time.time()
    result = run_backtest(cfg)
    elapsed = time.time() - start
    kpis = result['kpis']
    tfi_stats = result.get('meta', {}).get('tfi_stats', {})
    results.append({
        'scenario': name,
        'params': {**BASE_TDA, **overrides},
        'elapsed_sec': round(elapsed, 2),
        'kpis': kpis,
        'tfi_stats': tfi_stats,
    })
    print(json.dumps(results[-1], indent=2))

summary_path = 'artifacts/tda_sensitivity.json'
import pathlib
pathlib.Path('artifacts').mkdir(exist_ok=True)
pathlib.Path(summary_path).write_text(json.dumps(results, indent=2))
print(f'Wrote {summary_path}')