from __future__ import annotations

import pytest

from backtest.engine import _regime_target_vol


@pytest.mark.parametrize(
    "regime_value, expected_scale",
    [
        (0.0, 0.8),
        (0.5, 1.0),
        (1.0, 1.2),
    ],
)
def test_regime_target_vol_scaling(regime_value: float, expected_scale: float) -> None:
    base_target = 0.1
    target_vol, scale = _regime_target_vol(base_target, regime_value, cfg={})
    assert scale == pytest.approx(expected_scale)
    assert target_vol == pytest.approx(base_target * expected_scale)
