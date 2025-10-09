from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from features.tda.mapper import RegimeAwareMapper


def test_mapper_metrics_peripherality_and_centrality() -> None:
    """Ensure Mapper metrics produce finite centralities and peripherality scores."""
    mapper = RegimeAwareMapper()
    mapper.assets_ = ["AAA", "BBB", "CCC"]

    lens_series = pd.Series(
        data=[0.0, 0.5, 1.0], index=mapper.assets_, dtype=float
    )

    graph = nx.Graph()
    graph.add_node("0", members=[0], size=1)
    graph.add_node("1", members=[1], size=1)
    graph.add_node("2", members=[2], size=1)
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")

    metrics = mapper._compute_metrics(graph, lens_series)
    node_metrics = metrics.node_metrics

    assert metrics.summary["n_components"] == 1.0
    assert metrics.summary["gini_node_size"] == 0.0
    assert metrics.summary["avg_degree"] == pytest.approx(4.0 / 3.0)

    for column in (
        "degree_centrality",
        "eigenvector_centrality",
        "betweenness_centrality",
        "peripherality_score",
    ):
        assert column in node_metrics.columns
        assert node_metrics[column].notna().all()

    peripherality = node_metrics["peripherality_score"]
    assert peripherality.between(left=0.0, right=1.0).all()
    assert peripherality.loc["1"] < peripherality.loc["0"]
    assert peripherality.loc["1"] < peripherality.loc["2"]
