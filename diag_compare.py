from pathlib import Path
import json
import pandas as pd
from dataio.config import load_config
from dataio.loaders import get_panel
from src.backtest.engine import run_backtest
from src.validation.walk_forward import run_walk_forward

cfg = load_config("configs/base.yaml")
panel = get_panel(cfg["dates"]["start"], cfg["dates"]["end"])

backtest = run_backtest(cfg, panel)
walkforward = run_walk_forward(cfg, panel)

Path("reports/diag_bt_kpis.json").write_text(json.dumps(backtest["kpis"], indent=2))
Path("reports/diag_wf_kpis.json").write_text(json.dumps(walkforward["kpis"], indent=2))

print("Backtest:", backtest["kpis"])
print("Walk-forward:", walkforward["kpis"])
