import numpy as np
import pandas as pd
import pytest

from backtest.engine import softmax_with_temperature
from features import TFIParams, mapper_for_asset, tfi_score


def _synthetic_prices(n_assets: int = 2, n_days: int = 252, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    base = np.linspace(0, 6 * np.pi, n_days)
    data: dict[str, np.ndarray] = {}
    for i in range(n_assets):
        drift = 0.0004 * (i + 1)
        amplitude = 0.01 + 0.004 * i
        noise = rng.normal(scale=0.01, size=n_days)
        returns = drift + amplitude * np.sin(base + i) + noise
        prices = 100.0 * np.exp(np.cumsum(returns))
        data[f"A{i}"] = prices
    return pd.DataFrame(data, index=idx)


def test_mapper_for_asset_graph_metrics_sensitive_to_n_cubes():
    prices = _synthetic_prices()
    params_low = TFIParams(delay=1, dim=3, n_cubes=4, overlap=0.4, epsilon=None, min_samples=2, window=126)
    params_high = TFIParams(delay=1, dim=3, n_cubes=10, overlap=0.4, epsilon=None, min_samples=2, window=126)

    _, meta_low = mapper_for_asset(prices, params_low, "A0")
    _, meta_high = mapper_for_asset(prices, params_high, "A0")

    edge_low = meta_low["graph_metrics"]["edge_density"]
    edge_high = meta_high["graph_metrics"]["edge_density"]
    # Mudancas de n_cubes devem alterar a densidade media do grafo
    assert abs(edge_low - edge_high) > 1e-4


def test_mapper_for_asset_respects_end_date():
    prices = _synthetic_prices()
    params = TFIParams(delay=1, dim=3, n_cubes=6, overlap=0.5, window=126)
    end_date = prices.index[180]

    _, meta = mapper_for_asset(prices, params, "A1", end_date=end_date)

    assert meta["window"]["end"].startswith(str(end_date.date()))
    assert meta["window"]["length"] == params.window


def test_tfi_score_is_normalised_between_zero_and_one():
    prices = _synthetic_prices()
    params = TFIParams(delay=1, dim=3, n_cubes=6, overlap=0.6, window=126)

    scores = tfi_score(prices, params=params)

    assert not scores.empty
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    # Garantir que a serie tenha variabilidade suficiente para investigacoes
    assert scores.std(ddof=0) > 0.0


def test_softmax_temperature_behaviour():
    scores = pd.Series([0.2, 1.0, -0.5], index=["a", "b", "c"])

    hot = softmax_with_temperature(scores, temperature=2.0)
    cold = softmax_with_temperature(scores, temperature=0.2)
    uniform = pd.Series([1 / len(scores)] * len(scores), index=scores.index)

    # Temperaturas altas aproximam distribuicao uniforme
    assert (hot - uniform).abs().max() < 0.15
    # Temperaturas baixas criam contraste maior
    assert (cold.max() - cold.min()) > (hot.max() - hot.min())

    with pytest.raises(ValueError):
        softmax_with_temperature(scores, temperature=0.0)
