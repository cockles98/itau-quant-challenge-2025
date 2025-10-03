from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as tm

from features import forward_returns
from models.meta_blend import run_meta_blend


def test_forward_returns_basic():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 104.0, 106.0, 108.0],
            "BBB": [50.0, 50.0, 52.0, 54.0, 54.0],
        },
        index=dates,
    )
    result = forward_returns(prices, horizon=1)
    expected = prices.pct_change(periods=1).shift(-1)
    tm.assert_frame_equal(result, expected)


def test_meta_blend_produces_informative_scores():
    dates = pd.bdate_range("2020-01-01", periods=100)
    assets = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(7)

    momentum_data = {}
    quality_data = {}
    forward_data = {}
    for i, asset in enumerate(assets):
        base = np.sin(np.linspace(0, 6, len(dates))) + 0.1 * i
        quality = np.cos(np.linspace(0, 4, len(dates))) + 0.05 * i
        noise = rng.normal(0.0, 0.05, len(dates))
        target = 0.7 * base + 0.3 * quality + noise
        momentum_data[asset] = base
        quality_data[asset] = quality
        forward_data[asset] = target

    momentum_df = pd.DataFrame(momentum_data, index=dates)
    quality_df = pd.DataFrame(quality_data, index=dates)
    forward_df = pd.DataFrame(forward_data, index=dates)

    meta_cfg = {
        "enabled": True,
        "model_type": "ridge",
        "alpha_grid": [0.1],
        "cv_n_splits": 3,
        "cv_embargo_days": 1,
        "min_train_dates": 30,
        "rolling_window": 60,
        "standardize": True,
        "cv_metric": "ic",
    }

    scores, meta = run_meta_blend(
        momentum_df,
        quality_df,
        regime=None,
        forward_returns=forward_df,
        assets=assets,
        config_dict=meta_cfg,
    )

    assert not scores.empty
    assert "config" in meta

    combined = forward_df.loc[scores.index.intersection(forward_df.index)]
    ic_values = []
    for date in combined.index[-30:]:
        preds = scores.loc[date]
        actual = combined.loc[date]
        if preds.std(ddof=0) == 0 or actual.std(ddof=0) == 0:
            continue
        ic_values.append(preds.corr(actual))
    assert ic_values, "Expected non-empty IC sample"
    assert np.nanmean(ic_values) > 0.3
