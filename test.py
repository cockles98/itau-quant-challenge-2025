# 1. pip install pandas pyyaml
# 2. pip install -e .
# 3. definir caminho no 'powershell - TERMINAL' -> setx PYTHONPATH "$PWD\src"

from dataio import load_config; load_config('configs/base.yaml')
import pandas as pd
from dataio.loaders import get_panel, get_adv, select_universe

panel = get_panel('2024-01-01', '2024-06-30')
print(panel.head())

adv = get_adv(panel, lookback=20)
print(adv.dropna().head())

universe = select_universe(
    panel,
    top_n=3,
    adv_min=800_000,
    price_min=40,
    age_min=20,
    hysteresis_rebalances=2,
    calendar='W-FRI',
)
for dt, assets in list(universe.items())[:5]:
    print(dt.date(), assets)
