from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import mix_scores, peripherality_factor


def test_peripherality_factor_inverts_centrality() -> None:
    centrality = pd.Series({"A": 0.9, "B": 0.2, "C": 0.5}, dtype=float)
    factor = peripherality_factor(centrality, clip=5.0)

    assert isinstance(factor, pd.Series)
    assert factor.index.tolist() == ["A", "B", "C"]
    # Assets with lower centrality should get higher scores (more peripheral)
    assert factor["B"] > factor["C"] > factor["A"]
    # Z-score mean should be approximately zero
    assert np.isclose(float(factor.mean()), 0.0, atol=1e-8)


def test_mix_scores_with_peripherality_boost() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    assets = ["A", "B"]

    regime = pd.Series([0.8, 0.8], index=dates)
    momentum = pd.DataFrame([[1.0, 2.0], [1.5, 1.0]], index=dates, columns=assets)
    quality = pd.DataFrame([[0.5, 0.2], [0.4, 0.3]], index=dates, columns=assets)
    periph = pd.DataFrame([[0.6, -0.6], [0.7, -0.7]], index=dates, columns=assets)

    base = mix_scores(
        regime,
        momentum,
        quality,
        alpha=0.6,
        beta=0.3,
        gamma=0.1,
        regime_mode="linear",
    )
    boosted = mix_scores(
        regime,
        momentum,
        quality,
        alpha=0.6,
        beta=0.3,
        gamma=0.1,
        peripherality=periph,
        use_peripherality=True,
        delta=0.4,
        regime_mode="linear",
    )

    # Peripheral asset "A" receives an extra boost relative to baseline,
    # while central "B" is penalised.
    diff = boosted - base
    assert diff["mix_A"].iloc[0] > 0
    assert diff["mix_B"].iloc[0] < 0
