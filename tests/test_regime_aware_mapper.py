from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import RegimeAwareMapper


def _make_returns(rows: int = 80, cols: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = rng.normal(scale=0.01, size=(rows, cols))
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    columns = [f"A{i}" for i in range(cols)]
    return pd.DataFrame(data, index=index, columns=columns)


def _make_macros(index: pd.Index) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "selic": rng.normal(scale=0.0005, size=len(index)).cumsum(),
            "brlusd": rng.normal(scale=0.0008, size=len(index)).cumsum(),
        },
        index=index,
    )


def test_regime_mapper_basic(tmp_path) -> None:
    returns = _make_returns()
    macros = _make_macros(returns.index)

    mapper = RegimeAwareMapper(n_cubes=3, overlap=0.25, lens="pca_umap", min_cluster_size=2)
    mapper.fit(returns, macros=macros, regime_value=0.9)

    node_metrics = mapper.transform()
    assert not node_metrics.empty
    assert {"size", "members", "degree_centrality"}.issubset(node_metrics.columns)
    assert mapper.metrics_ is not None
    assert set(mapper.metrics_.summary.keys()) == {"n_components", "avg_degree", "gini_node_size"}
    assert mapper.cover_params_ is not None
    n_cubes, overlap = mapper.cover_params_
    assert n_cubes >= mapper.base_n_cubes
    assert overlap >= mapper.base_overlap

    json_path, png_path = mapper.export_graph(tmp_path / "mapper_graph")
    assert json_path.exists()
    assert png_path.exists()


def test_regime_mapper_macro_fallback() -> None:
    returns = _make_returns(rows=60, cols=4)
    mapper = RegimeAwareMapper(lens="beta_selic")
    mapper.fit(returns, macros=None)

    assert mapper.lens_values_ is not None
    assert mapper.lens_values_.name in {"lens_volatility", "lens_sharpe"}
