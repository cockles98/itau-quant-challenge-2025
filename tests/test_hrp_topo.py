from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from portfolio import topo_seriation_from_graph


def _sample_cov() -> pd.DataFrame:
    assets = ["A", "B", "C", "D"]
    cov_values = np.array(
        [
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.5, 0.4, 0.2],
            [0.2, 0.4, 1.2, 0.5],
            [0.1, 0.2, 0.5, 2.0],
        ],
        dtype=float,
    )
    return pd.DataFrame(cov_values, index=assets, columns=assets)


def test_mapper_seriation_respects_centrality() -> None:
    cov = _sample_cov()
    base_graph = nx.path_graph(cov.columns)

    mapper = nx.Graph()
    mapper.add_node("cluster1", members=["A"], centrality=10.0)
    mapper.add_node("cluster2", members=["B", "C"], centrality=5.0)
    mapper.add_node("cluster3", members=["D"], centrality=1.0)
    mapper.add_edges_from(
        [
            ("cluster1", "cluster2"),
            ("cluster2", "cluster3"),
        ]
    )

    order = topo_seriation_from_graph(cov, base_graph, mapper_graph=mapper)

    assert order[:4] == ["A", "B", "C", "D"]
    assert set(order) == set(cov.columns)


def test_seriation_falls_back_without_mapper() -> None:
    cov = _sample_cov()
    base_graph = nx.path_graph(cov.columns)

    order = topo_seriation_from_graph(cov, base_graph)

    assert len(order) == len(cov.columns)
    assert set(order) == set(cov.columns)


def test_mapper_members_int_indices() -> None:
    cov = _sample_cov()
    base_graph = nx.path_graph(cov.columns)

    mapper = nx.Graph()
    mapper.add_node("node0", members=[0, 1], centrality=3.0)
    mapper.add_node("node1", members=[2], centrality=1.0)
    mapper.add_edges_from([("node0", "node1")])

    order = topo_seriation_from_graph(cov, base_graph, mapper_graph=mapper)

    assert order[0] in {"A", "B"}
    assert set(order) == set(cov.columns)
