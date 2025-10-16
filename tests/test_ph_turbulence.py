from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import PHTurbulenceTransformer


def _synthetic_returns(rows: int = 80, cols: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(scale=0.01, size=(rows, cols))
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    columns = [f"TICK{i}" for i in range(cols)]
    return pd.DataFrame(data, index=index, columns=columns)


def test_turbulence_series_alignment() -> None:
    df = _synthetic_returns()
    transformer = PHTurbulenceTransformer(window=20, compute_wasserstein=True)
    transformer.fit(df)
    series = transformer.transform(df)

    assert isinstance(series, pd.Series)
    assert series.index.equals(df.index)
    assert series.name == "ph_turbulence"
    assert series.isna().sum() == transformer.window - 1
    assert series.iloc[transformer.window - 1 :].notna().all()


def test_extra_features_and_meta() -> None:
    df = _synthetic_returns()
    transformer = PHTurbulenceTransformer(window=16, compute_wasserstein=True)
    transformer.fit(df)
    _ = transformer.transform(df)

    assert transformer.extra_features_ is not None
    extras = transformer.extra_features_
    assert set(["persistence_entropy", "amplitude_sum", "wasserstein_shift"]).issubset(
        extras.columns
    )
    # Wasserstein shift is undefined for the first valid window.
    valid_extras = extras.iloc[transformer.window - 1 :]
    assert valid_extras["persistence_entropy"].notna().any()
    assert valid_extras["amplitude_sum"].notna().any()
    assert valid_extras["wasserstein_shift"].iloc[1:].notna().any()

    assert transformer.meta_ is not None
    assert "diag_count_H0" in transformer.meta_
    assert "diag_count_H1" in transformer.meta_
    h0 = transformer.meta_["diag_count_H0"]
    h1 = transformer.meta_["diag_count_H1"]
    assert isinstance(h0, pd.Series) and isinstance(h1, pd.Series)
    assert h0.index.equals(df.index)
    assert h1.index.equals(df.index)
    assert h0.notna().sum() > 0
    assert h1.notna().sum() > 0


def test_set_params_changes_configuration() -> None:
    transformer = PHTurbulenceTransformer()
    updated = transformer.set_params(norm="l1", compute_entropy=False)
    assert updated is transformer
    assert transformer.norm == "l1"
    assert transformer.compute_entropy is False
