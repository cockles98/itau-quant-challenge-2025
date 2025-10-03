from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as tm

from portfolio import rolling_cov


def _manual_ewma_cov(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    n_obs = values.shape[0]
    weights = lam ** np.arange(n_obs)[::-1]
    weights = weights / weights.sum()
    mean = np.average(values, axis=0, weights=weights)
    demeaned = values - mean
    weighted = demeaned * weights[:, None]
    cov = weighted.T @ demeaned
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def test_rolling_cov_with_ewma_matches_manual_calculation():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03, 0.04],
            "BBB": [-0.02, 0.01, 0.00, 0.02, 0.05],
        },
        index=dates,
    )

    lam = 0.5
    cov_dict = rolling_cov(returns, window=4, method_cfg={"ewma_lambda": lam})
    cov = cov_dict[dates[-1]]
    manual = _manual_ewma_cov(returns.iloc[-4:], lam)
    tm.assert_frame_equal(cov, manual, check_exact=False, rtol=1e-10, atol=1e-12)


def test_shrinkage_diagonal_zeroes_off_diagonals_when_strength_one():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03],
            "BBB": [-0.02, 0.01, 0.00, 0.02],
        },
        index=dates,
    )

    cov_dict = rolling_cov(
        returns,
        window=4,
        method_cfg={"shrinkage": "diagonal", "shrinkage_strength": 1.0},
    )
    cov = cov_dict[dates[-1]]
    assert np.isclose(cov.loc["AAA", "BBB"], 0.0)
    assert np.isclose(cov.loc["BBB", "AAA"], 0.0)
    assert cov.loc["AAA", "AAA"] > 0
    assert cov.loc["BBB", "BBB"] > 0
