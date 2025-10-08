from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.regime.ph_regime import compute_ph_regime_index


def _dummy_returns(rows: int = 60, cols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = rng.normal(scale=0.01, size=(rows, cols))
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    columns = [f"T{i}" for i in range(cols)]
    return pd.DataFrame(data, index=index, columns=columns)


def test_disabled_config_returns_zero_index() -> None:
    returns = _dummy_returns()
    cfg = {"tda_ph": {"enabled": False}}

    series = compute_ph_regime_index(returns, cfg)

    assert isinstance(series, pd.Series)
    assert series.eq(0.0).all()
    assert "is_alert" in series.attrs
    assert "is_riskoff" in series.attrs
    assert not series.attrs["is_alert"].any()
    assert not series.attrs["is_riskoff"].any()


def test_riskoff_flag_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = _dummy_returns()
    base = pd.Series(0.0, index=returns.index, name="ph_turbulence")
    base.iloc[-3:] = [0.0, 0.0, 50.0]

    monkeypatch.setattr(
        "src.features.regime.ph_regime._compute_base_series",
        lambda _returns, _cfg: base,
    )

    cfg = {
        "tda_ph": {
            "enabled": True,
            "window": 10,
            "homology_dim": 1,
            "norm": "l2",
            "smooth_span": 1,
            "zscore_lookback": 3,
            "alert_sigma": 0.5,
            "riskoff_sigma": 1.0,
        }
    }

    series = compute_ph_regime_index(returns, cfg)

    assert series.iloc[-1] == pytest.approx(1.0, abs=1e-6)
    alert_flags = series.attrs["is_alert"]
    riskoff_flags = series.attrs["is_riskoff"]
    assert bool(riskoff_flags.iloc[-1])
    assert bool(alert_flags.iloc[-1])
    assert series.iloc[-4] == pytest.approx(0.0, abs=1e-8)
