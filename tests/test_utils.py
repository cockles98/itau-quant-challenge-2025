import numpy as np
import pandas as pd

from dataio.loaders import select_universe
from features import momentum_12_1
from risk.risk_controls import apply_turnover_cap, kill_switch


def _make_panel(days: int = 40):
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    assets = ["AAA", "BBB"]
    records = []
    for asset in assets:
        price = 100 + np.arange(len(dates))
        volume = np.full(len(dates), 1_000_000)
        for dt, p, v in zip(dates, price, volume):
            records.append((dt, asset, p, v))
    return pd.DataFrame(
        records, columns=["date", "asset", "close", "volume"]
    ).set_index(["date", "asset"])


def test_universe_hysteresis():
    panel = _make_panel()
    universe = select_universe(
        panel,
        top_n=1,
        adv_min=0,
        price_min=0,
        age_min=1,
        hysteresis_rebalances=2,
        calendar="B",
    )
    ordered_dates = sorted(universe)
    non_empty = [d for d in ordered_dates if universe[d]]
    assert non_empty, "Universe never selects any asset"
    first_date = non_empty[0]
    first_idx = ordered_dates.index(first_date)
    if first_idx > 0:
        assert universe[ordered_dates[first_idx - 1]] == []


def test_factor_no_leak():
    prices = pd.DataFrame(
        {"AAA": np.linspace(100, 150, 400), "BBB": np.linspace(50, 80, 400)},
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    mom = momentum_12_1(prices)
    assert not mom.empty
    earliest = mom.index.min()
    assert (
        earliest >= prices.index[252]
    ), "Momentum should drop early rows to avoid look-ahead"


def test_turnover_cap():
    prev = pd.Series({"AAA": 0.5, "BBB": 0.5})
    new = pd.Series({"AAA": 1.0, "BBB": 0.0})
    cap = 0.25
    capped = apply_turnover_cap(prev, new, cap=cap)
    turnover = 0.5 * (capped - prev).abs().sum()
    assert turnover <= cap + 1e-9
    assert abs(capped.sum() - 1.0) < 1e-9


def test_kill_switch_triggers_on_drawdown():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    equity = pd.Series(np.linspace(1.0, 0.6, len(dates)), index=dates)
    vol = pd.Series(0.15, index=dates)
    assert kill_switch(equity, vol, vol_target=0.1, mdd_thres=-0.1)


def test_kill_switch_respects_vol_target():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    equity = pd.Series(1.0, index=dates)
    vol = pd.Series(0.25, index=dates)
    assert kill_switch(equity, vol, vol_target=0.1, vol_mult=2.0)
