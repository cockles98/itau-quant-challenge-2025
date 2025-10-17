
import pandas as pd

from backtest.engine import run_backtest


def _synthetic_panel(days: int = 260):
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    assets = ["AAA", "BBB", "CCC"]
    records = []
    for idx, date in enumerate(dates):
        for j, asset in enumerate(assets):
            price = 100.0 + 0.2 * idx + 0.5 * j
            volume = 1_000_000.0 + 5_000.0 * j
            records.append((date, asset, price, volume))
    panel = pd.DataFrame(records, columns=["date", "asset", "close", "volume"])
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.set_index(["date", "asset"])


def _base_config(start: str, end: str) -> dict:
    return {
        "seeds": 0,
        "data": {"source": "synthetic"},
        "paths": {"data": ".", "reports": ".", "artifacts": "."},
        "dates": {"start": start, "end": end},
        "portfolio": {},
        "tda_ph": {
            "enabled": True,
            "window": 30,
            "homology_dim": 1,
            "norm": "l2",
            "smooth_span": 10,
            "zscore_lookback": 120,
            "alert_sigma": 0.6,
            "riskoff_sigma": 1.9,
        },
        "mapper": {
            "enabled": True,
            "lens": "pca_umap",
            "n_cubes": 6,
            "overlap": 0.5,
            "eps_quantile": 0.2,
            "min_cluster_size": 6,
            "epsilon_adaptive": True,
        },
        "factors": {
            "alphas": ["momentum", "quality", "carry"],
            "regime_gain": 1.0,
            "alpha": 0.5,
            "beta": 0.3,
            "gamma": 0.1,
            "delta": 0.1,
            "use_peripherality": True,
            "softmax_T": 1.0,
        },
        "windows": {"vol_window": 40, "atr_len": 20},
        "costs": {"fee_bps": 0.0, "k": 0.0, "max_bps": 50.0},
        "participation_cap": 0.5,
        "turnover_cap": 1.0,
        "vol_target": 0.2,
        "universe": {
            "top_n": 3,
            "adv_min": 0.0,
            "price_min": 0.0,
            "age_min": 5,
            "hysteresis_rebalances": 1,
            "calendar": "B",
        },
        "risk": {
            "mdd_lookback": 40,
            "mdd_thres": -0.3,
            "vol_mult": 3.0,
            "cooldown_days": 5,
            "reentry_hysteresis": 0.05,
        },
    }


def _hrp_only_config(start: str, end: str) -> dict:
    cfg = _base_config(start, end)
    portfolio = cfg.setdefault("portfolio", {})
    portfolio["method"] = "hrp_only"
    portfolio.setdefault("hrp_only_regime_value", 0.5)
    return cfg


def _tda_only_config(start: str, end: str) -> dict:
    cfg = _base_config(start, end)
    cfg.setdefault("portfolio", {})["method"] = "tda_only"
    return cfg


def test_run_backtest_hrp_only_sets_constant_regime():
    panel = _synthetic_panel()
    dates = panel.index.get_level_values("date")
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    cfg = _hrp_only_config(start, end)

    result = run_backtest(cfg, panel=panel)

    assert not result["equity_curve"].isna().any()
    portfolio_meta = result["meta"].get("portfolio", {})
    assert portfolio_meta.get("method") == "hrp_only"
    assert portfolio_meta.get("regime_constant") == 0.5
    assert portfolio_meta.get("base_allocation") == "hrp"

    tda_meta = result["meta"].get("tda_params", {})
    assert tda_meta.get("mode") == "hrp_only"
    regime_stats = result["meta"].get("regime_stats", {})
    assert regime_stats.get("std") == 0.0


def test_run_backtest_tda_only_sets_uniform_base_allocation():
    panel = _synthetic_panel()
    dates = panel.index.get_level_values("date")
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    cfg = _tda_only_config(start, end)

    result = run_backtest(cfg, panel=panel)

    assert not result["equity_curve"].isna().any()
    portfolio_meta = result["meta"].get("portfolio", {})
    assert portfolio_meta.get("method") == "tda_only"
    assert portfolio_meta.get("base_allocation") == "uniform"

    tda_meta = result["meta"].get("tda_params", {})
    assert tda_meta.get("mode") == "tda_only"

