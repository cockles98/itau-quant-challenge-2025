# 1. pip install pandas pyyaml
# 2. pip install -e .
# 3. definir caminho no "powershell - TERMINAL" -> setx PYTHONPATH "$PWD\src"

import numpy as np
import pandas as pd
import networkx as nx

from backtest import commission_cost, execute_trade, slippage_cost
from backtest.engine import run_backtest
from dataio import load_config
from dataio.loaders import get_adv, get_panel, select_universe
from features import (
    TFIParams,
    mapper_graph,
    mix_scores,
    momentum_12_1,
    quality_proxy,
    slope_nd,
    takens_embedding,
    tfi_score,
)
from portfolio import hrp_weights_from_order, rolling_cov, topo_seriation_from_graph
from risk import atr, atr_risk_normalize, scale_to_vol
from risk import risk_controls
from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, turnover, vol
from reports import plot_equity_curves, table_kpis
from validation import (
    run_walk_forward,
    purged_kfold_split,
    tune_params,
    param_sensitivity_heatmaps,
    stress_costs,
    regime_subperiods,
    capacity_curve,
)

cfg = load_config("configs/base.yaml")

panel = get_panel("2024-01-01", "2024-06-30")
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
    calendar="W-FRI",
)
for dt, assets in list(universe.items())[:5]:
    print(dt.date(), assets)

# Custos: comparar duas execucoes com a mesma ordem, mas diferentes ADVs
notional = 100_000
fee_bps = 5

commission = commission_cost(notional, fee_bps)
print(f"Commission for 100k @ {fee_bps} bps: {commission:.2f}")

slip_high_adv = slippage_cost(notional, adv=5_000_000, k=0.1, max_bps=50)
slip_low_adv = slippage_cost(notional, adv=500_000, k=0.1, max_bps=50)
print(f"Slippage with ADV=5M: {slip_high_adv:.2f}")
print(f"Slippage with ADV=500k: {slip_low_adv:.2f}")

trade_high_adv = execute_trade(
    prev_qty=0,
    target_qty=10_000,
    price=10,
    adv=5_000_000,
    fee_bps=fee_bps,
    slip_params={"k": 0.1, "max_bps": 50},
)
trade_low_adv = execute_trade(
    prev_qty=0,
    target_qty=10_000,
    price=10,
    adv=500_000,
    fee_bps=fee_bps,
    slip_params={"k": 0.1, "max_bps": 50},
)

print("High ADV trade:", trade_high_adv)
print("Low ADV trade:", trade_low_adv)

# TDA: Takens embedding, Mapper graph e TFI score
np.random.seed(42)
synthetic_series = np.sin(np.linspace(0, 10, 100))
embedded = takens_embedding(synthetic_series, delay=2, dim=3)
print("Takens embedding shape:", embedded.shape)

mapper = mapper_graph(embedded, n_cubes=4, overlap=0.5)
print("Mapper nodes/edges:", mapper.number_of_nodes(), mapper.number_of_edges())

price_index = pd.date_range("2024-01-01", periods=120, freq="B")
price_frame = pd.DataFrame(
    {
        "AAA": 100
        * np.exp(np.cumsum(0.0002 + 0.01 * np.random.randn(len(price_index)))),
        "BBB": 120
        * np.exp(np.cumsum(-0.0001 + 0.012 * np.random.randn(len(price_index)))),
    },
    index=price_index,
)
params = TFIParams(window=40, delay=1, dim=3, n_cubes=4, overlap=0.5)
regime_score = tfi_score(price_frame, params=params)
print("TFI score tail:\n", regime_score.tail())

# Fatores canonicos e mistura
long_index = pd.date_range("2023-01-02", periods=400, freq="B")
long_prices = pd.DataFrame(
    {
        "AAA": 100
        * np.exp(np.cumsum(0.0003 + 0.012 * np.random.randn(len(long_index)))),
        "BBB": 90
        * np.exp(np.cumsum(0.0001 + 0.011 * np.random.randn(len(long_index)))),
        "CCC": 110
        * np.exp(np.cumsum(-0.0002 + 0.013 * np.random.randn(len(long_index)))),
    },
    index=long_index,
)

mom_scores = momentum_12_1(long_prices)
slope_scores = slope_nd(long_prices, n=63)
qual_scores = quality_proxy(long_prices)
regime_long = tfi_score(long_prices, params=TFIParams(window=80))

mix = mix_scores(regime_long, mom_scores, qual_scores, alpha=0.6, beta=0.3, gamma=0.1)
print("Momentum head:\n", mom_scores.dropna().head())
print("Slope head:\n", slope_scores.dropna().head())
print("Quality head:\n", qual_scores.dropna().head())
print("Mixed head:\n", mix.dropna().head())

# HRP pipeline quick test
returns_index = pd.date_range("2024-01-01", periods=180, freq="B")
returns = pd.DataFrame(
    0.001 * np.random.randn(len(returns_index), 5),
    index=returns_index,
    columns=["AAA", "BBB", "CCC", "DDD", "EEE"],
)

cov_dict = rolling_cov(returns, window=60)
last_date, last_cov = next(reversed(cov_dict.items()))
print("Rolling cov last date:", last_date)

asset_graph = nx.path_graph(last_cov.columns)
order = topo_seriation_from_graph(last_cov, asset_graph)
print("HRP order:", order)

weights = hrp_weights_from_order(last_cov, order)
print("HRP weights:\n", weights)
print("Weights sum:", weights.sum())

# Risk normalisation and scaling
risk_index = pd.date_range("2024-01-01", periods=200, freq="B")
noise = 0.001 * np.random.randn(len(risk_index))
risk_returns = pd.DataFrame(
    {
        "AAA": 0.005 + noise,
        "BBB": 0.010 + 2 * noise,
    },
    index=risk_index,
)
risk_prices = 100 * (1 + risk_returns).cumprod()

atr_values = atr(risk_prices, n=14)
print("ATR tail:\n", atr_values.tail())
print("ATR ratio BBB/AAA:", (atr_values["BBB"] / atr_values["AAA"]).iloc[-1])

raw_weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
atr_scaled = atr_risk_normalize(raw_weights, atr_values)
print("ATR-normalised weights:", atr_scaled)
print("Weight ratio AAA/BBB:", atr_scaled["AAA"] / atr_scaled["BBB"])

risk_cov_dict = rolling_cov(risk_returns, window=60)
_, risk_cov = next(reversed(risk_cov_dict.items()))
scaled_weights = scale_to_vol(atr_scaled, risk_returns, target_vol=0.10, window=60)
print("Scaled weights:", scaled_weights)
active_assets = scaled_weights.index.intersection(risk_cov.index)
scaled_vol = np.sqrt(
    float(
        scaled_weights.loc[active_assets]
        @ risk_cov.loc[active_assets, active_assets]
        @ scaled_weights.loc[active_assets]
    )
)
print("Achieved vol:", scaled_vol)

# Risk controls
weights = pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1})
clusters = {
    "cluster_1": ["AAA", "BBB"],
    "cluster_2": ["CCC", "DDD"],
}

capped_weights = risk_controls.apply_caps(
    weights, max_asset=0.35, max_cluster=0.55, clusters=clusters
)
print("Capped weights:\n", capped_weights)
print("Sum after caps:", capped_weights.sum())

prev_w = pd.Series({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25})
new_w = pd.Series({"AAA": 0.50, "BBB": 0.20, "CCC": 0.20, "DDD": 0.10})
adjusted_w = risk_controls.apply_turnover_cap(prev_w, new_w, cap=0.25)
print("Turnover capped weights:", adjusted_w)

idx = pd.date_range("2024-01-01", periods=150, freq="B")
equity = pd.Series(100 * (1 + 0.001 * np.random.randn(len(idx))).cumprod(), index=idx)
vol_series = pd.Series(0.12 + 0.02 * np.random.randn(len(idx)), index=idx)
ks_trigger = risk_controls.kill_switch(
    equity, vol_series, vol_target=0.10, mdd_lookback=60, mdd_thres=-0.15, vol_mult=1.5
)
print("Kill switch triggered:", ks_trigger)

# Full backtest run
engine_result = run_backtest(cfg)
print("Backtest final equity:", engine_result["kpis"]["final_equity"])

# KPI metrics and reporting
np.random.seed(7)
metric_index = pd.date_range("2024-01-01", periods=252, freq="B")
metric_returns = pd.Series(
    np.random.normal(0.0005, 0.01, len(metric_index)), index=metric_index
)
metric_equity = (1 + metric_returns).cumprod()
weights_df = pd.DataFrame(
    {
        "AAA": 0.5 + 0.1 * np.sin(np.linspace(0, 6.28, len(metric_index))),
        "BBB": 0.5 - 0.1 * np.sin(np.linspace(0, 6.28, len(metric_index))),
    },
    index=metric_index,
)
turnover_series = turnover(weights_df)
print("Sharpe:", sharpe(metric_returns))
print("Sortino:", sortino(metric_returns))
print("CAGR:", cagr(metric_equity))
print("Vol:", vol(metric_returns))
print("Max DD:", mdd(metric_equity))
print("Calmar:", calmar(metric_equity))
print("Hit rate:", hit_rate(metric_returns))
print("Turnover mean:", turnover_series.mean())

curves_path = plot_equity_curves(
    {"Strategy": metric_equity, "Benchmark": metric_equity * 0.98}
)
print("Equity plot saved to:", curves_path)

results_table = table_kpis(
    {
        "Strategy": {
            "equity_curve": metric_equity,
            "returns": metric_returns,
            "weights": weights_df,
        },
        "Benchmark": {
            "equity_curve": metric_equity * 0.98,
            "returns": metric_returns * 0.8,
        },
    }
)
print("KPI table:\n", results_table)
# Walk-forward validation
np.random.seed(21)
wf_dates = pd.date_range("2020-01-01", "2022-12-31", freq="B")
wf_assets = ["AAA", "BBB", "CCC"]
records = []
for asset in wf_assets:
    path = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(wf_dates))))
    volume = np.abs(1_000_000 + np.random.normal(0, 100_000, len(wf_dates)))
    asset_df = pd.DataFrame(
        {"date": wf_dates, "asset": asset, "close": path, "volume": volume}
    )
    records.append(asset_df)
walk_panel = pd.concat(records).set_index(["date", "asset"]).sort_index()
wf_result = run_walk_forward(cfg, walk_panel)
print("Walk-forward equity tail:\n", wf_result["equity_curve"].tail())
print("Walk-forward KPIs:", wf_result["kpis"])
print("Walk-forward windows:", len(wf_result["windows"]))


# Purged K-Fold tuning
cv_dates = pd.date_range("2020-01-01", "2020-06-30", freq="B")
assets_cv = ["AAA", "BBB", "CCC"]
cv_records = []
np.random.seed(123)
for asset in assets_cv:
    price_path = 100 * np.exp(np.cumsum(np.random.normal(0, 0.012, len(cv_dates))))
    volume_series = np.abs(1_000_000 + np.random.normal(0, 80_000, len(cv_dates)))
    cv_records.append(
        pd.DataFrame(
            {
                "date": cv_dates,
                "asset": asset,
                "close": price_path,
                "volume": volume_series,
            }
        )
    )
cv_panel = pd.concat(cv_records).set_index(["date", "asset"]).sort_index()

splits = purged_kfold_split(cv_dates, n_splits=3, embargo_days=5)
print("Purged splits lens:", [(len(tr), len(te)) for tr, te in splits])

param_grid = {
    "delay": [1],
    "dim": [3],
    "n_cubes": [4],
    "overlap": [0.5],
    "alpha": [0.6],
    "beta": [0.3],
    "gamma": [0.1],
}
cfg_purged = cfg.copy()
cfg_purged.setdefault("dates", {})
cfg_purged["dates"]["start"] = cv_dates[0].isoformat()
cfg_purged["dates"]["end"] = cv_dates[-1].isoformat()

best_params, grid_results = tune_params(
    cfg_purged, cv_panel, param_grid, n_splits=2, embargo_days=5
)
print("Purged tuning best params:", best_params)
print("Purged tuning Sharpe:", grid_results["sharpe"].tolist())

# Robustness and stress testing
heat_grid = {"delay": [1, 2], "dim": [2, 3]}
heatmap_paths = param_sensitivity_heatmaps(cfg_purged, cv_panel, heat_grid)
print("Heatmap files:", [str(p) for p in heatmap_paths])

stress_path = stress_costs(cfg_purged, cv_panel, multipliers=[0.8, 1.0])
print("Stress costs saved to:", stress_path)

tfi_series = pd.Series(np.linspace(-1, 1, len(cv_dates)), index=cv_dates)
regime_table = regime_subperiods(cfg_purged, cv_panel, tfi_series)
print("Regime subperiod KPIs:\n", regime_table)
capacity_outputs = capacity_curve(cfg_purged, cv_panel, participation_caps=[0.01, 0.05])
print("Capacity outputs:", {k: str(v) for k, v in capacity_outputs.items()})
