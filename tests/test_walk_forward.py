import pandas as pd
import numpy as np
from unittest.mock import patch

from validation.walk_forward import run_walk_forward


def _make_panel(days: int = 756):
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    assets = ["AAA", "BBB"]
    records = []
    for dt in dates:
        for asset in assets:
            records.append((dt, asset, 100.0, 1_000_000.0))
    return pd.DataFrame(records, columns=["date", "asset", "close", "volume"]).set_index(["date", "asset"])


def test_run_walk_forward_respects_max_windows():
    panel = _make_panel(days=252 * 3)
    cfg = {}

    def fake_backtest(cfg_window, panel=None):
        dates = panel.index.get_level_values(0).unique().sort_values()
        equity = pd.Series(np.linspace(1.0, 1.1, len(dates)), index=dates)
        weights = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
        return {
            "equity_curve": equity,
            "daily_positions": weights,
            "kpis": {"Sharpe": 1.0},
        }

    with patch("validation.walk_forward.run_backtest", side_effect=fake_backtest):
        result = run_walk_forward(cfg, panel, max_windows=1)

    assert len(result["windows"]) == 1
    assert result["returns"].index.min() >= result["windows"][0]["oos_start"]
