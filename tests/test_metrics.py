import pandas as pd
import pytest

from metrics import avg_time_under_water, max_time_under_water


def test_time_under_water_metrics():
    equity = pd.Series([1.0, 0.9, 0.8, 1.0, 1.1, 0.9, 1.1])
    assert avg_time_under_water(equity) == pytest.approx(1.5)
    assert max_time_under_water(equity) == pytest.approx(2.0)


def test_time_under_water_zero_when_no_drawdown():
    equity = pd.Series([1.0, 1.0, 1.0])
    assert avg_time_under_water(equity) == 0.0
    assert max_time_under_water(equity) == 0.0
