from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = ["takens_embedding", "mapper_graph", "tfi_score", "TFIParams"]


@dataclass(frozen=True)
class TFIParams:
    delay: int = 1
    dim: int = 3
    n_cubes: int = 5
    overlap: float = 0.5
    epsilon: Optional[float] = None
    min_samples: int = 2
    window: int = 63


def takens_embedding(series: Iterable[float], delay: int, dim: int) -> np.ndarray:
    """Construct a Takens embedding of the input series."""
    data = np.asarray(series, dtype=float)
    if delay <= 0:
        raise ValueError("delay must be positive")
    if dim <= 1:
        raise ValueError("dim must be at least 2")
    if len(data) < (dim - 1) * delay + 1:
        raise ValueError("series length insufficient for requested embedding")

    n_vectors = len(data) - (dim - 1) * delay
    return np.column_stack(
        [data[i : i + n_vectors] for i in range(0, dim * delay, delay)]
    )


def mapper_graph(
    embedded: np.ndarray,
    n_cubes: int,
    overlap: float,
    metric: str = "euclidean",
    epsilon: Optional[float] = None,
    min_samples: int = 2,
) -> nx.Graph:
    """Construct a Mapper-like graph using coarse binning and DBSCAN clusters."""

    if embedded.ndim != 2:
        raise ValueError("embedded must be 2D array")
    if embedded.shape[0] < 2:
        raise ValueError("embedded must contain at least two samples")
    if n_cubes <= 0:
        raise ValueError("n_cubes must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    embedded = np.asarray(embedded, dtype=float)
    scaler = StandardScaler()
    scaled_embedded = scaler.fit_transform(embedded)

    # Project onto first principal component to define lens
    lens = scaled_embedded[:, 0]
    min_val, max_val = lens.min(), lens.max()
    if max_val == min_val:
        max_val += 1e-9

    cube_size = (max_val - min_val) / n_cubes
    overlap_th = cube_size * overlap

    # Assign points to overlapping intervals
    intervals = []
    for cube in range(n_cubes):
        left = min_val + cube * cube_size - overlap_th
        right = min_val + (cube + 1) * cube_size + overlap_th
        mask = (lens >= left) & (lens <= right)
        if mask.sum() == 0:
            continue
        intervals.append((cube, scaled_embedded[mask]))

    graph = nx.Graph()
    cluster_id = 0

    for cube_idx, points in intervals:
        eps = epsilon if epsilon is not None else cube_size / 2
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clustering.fit_predict(points)
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            node_points = points[labels == label]
            node_name = f"{cube_idx}_{cluster_id}"
            graph.add_node(node_name, cube=cube_idx, size=len(node_points))
            cluster_id += 1

    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if graph.nodes[nodes[i]]["cube"] == graph.nodes[nodes[j]]["cube"]:
                graph.add_edge(nodes[i], nodes[j])

    return graph


def _graph_features(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {"avg_degree": 0.0, "clustering": 0.0, "components": 0.0}
    avg_degree = np.mean([deg for _, deg in graph.degree()])
    clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0.0
    components = nx.number_connected_components(graph)
    return {
        "avg_degree": float(avg_degree),
        "clustering": float(clustering),
        "components": float(components),
    }


def _normalise_features(features: Dict[str, float]) -> float:
    vals = np.array(list(features.values()), dtype=float)
    if vals.ptp() == 0:
        return float(vals.mean()) if vals.size else 0.0
    return float((vals - vals.min()) / (vals.max() - vals.min()).mean())


def tfi_score(prices: pd.DataFrame, params: TFIParams | None = None) -> pd.Series:
    if params is None:
        params = TFIParams()

    if prices.isnull().any().any():
        prices = prices.ffill()
        prices = prices.dropna(axis=1, how="all")

    scores = {}
    for end_idx in range(params.window, len(prices) + 1):
        window_prices = prices.iloc[end_idx - params.window : end_idx]
        log_returns = np.log(window_prices).diff()
        features = []
        for column in log_returns:
            series = log_returns[column].dropna().to_numpy()
            if len(series) < (params.dim - 1) * params.delay + 1:
                continue
            try:
                embedded = takens_embedding(
                    series,
                    delay=params.delay,
                    dim=params.dim,
                )
                if embedded.shape[0] < 2:
                    continue
                graph = mapper_graph(
                    embedded,
                    params.n_cubes,
                    params.overlap,
                    epsilon=params.epsilon,
                    min_samples=params.min_samples,
                )
                features.append(_graph_features(graph))
            except ValueError as exc:
                logger.debug("Skipping series %s due to error: %s", column, exc)
        if not features:
            score = 0.0
        else:
            avg_features = {
                key: float(np.mean([feat[key] for feat in features]))
                for key in features[0]
            }
            vals = np.array(list(avg_features.values()), dtype=float)
            if np.all(vals == vals[0]):
                score = float(vals[0])
            else:
                min_val, max_val = vals.min(), vals.max()
                score = float((vals.mean() - min_val) / (max_val - min_val))
        scores[window_prices.index[-1]] = max(0.0, min(1.0, score))

    return pd.Series(scores, name="tfi_score")
