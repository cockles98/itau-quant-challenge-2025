from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from umap import UMAP

logger = logging.getLogger(__name__)


@dataclass
class MapperMetrics:
    summary: dict[str, float]
    node_metrics: pd.DataFrame
    graph: nx.Graph


class RegimeAwareMapper:
    """KeplerMapper workflow that adapts resolution according to market regimes."""

    def __init__(
        self,
        *,
        n_cubes: int = 6,
        overlap: float = 0.35,
        lens: str = "pca_umap",
        min_cluster_size: int = 3,
        eps_quantile: float = 0.2,
        epsilon_adaptive: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if n_cubes <= 0:
            raise ValueError("n_cubes must be positive.")
        if not 0.0 < overlap < 1.0:
            raise ValueError("overlap must lie in (0, 1).")
        valid_lenses = {
            "pca_umap",
            "volatility",
            "beta_selic",
            "beta_usd",
            "beta_ipca",
            "custom",
        }
        if lens not in valid_lenses:
            raise ValueError(f"lens must be one of {sorted(valid_lenses)}.")
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1.")
        if not 0.0 < eps_quantile <= 1.0:
            raise ValueError("eps_quantile must be in (0, 1].")

        self.base_n_cubes = int(n_cubes)
        self.base_overlap = float(overlap)
        self.lens = lens
        self.min_cluster_size = int(min_cluster_size)
        self.eps_quantile = float(eps_quantile)
        self.epsilon_adaptive = bool(epsilon_adaptive)
        self.random_state = random_state

        self.mapper_ = km.KeplerMapper(verbose=0)
        self.cover_params_: tuple[int, float] | None = None
        self.metrics_: MapperMetrics | None = None
        self.graph_: nx.Graph | None = None
        self.assets_: list[str] | None = None
        self.lens_values_: pd.Series | None = None
        self._graph_payload: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    def fit(
        self,
        returns: pd.DataFrame,
        *,
        macros: pd.DataFrame | None = None,
        regime_value: float | None = None,
        custom_lens: pd.Series | None = None,
    ) -> "RegimeAwareMapper":
        """Compute Mapper graph from asset-level return matrix."""
        returns = self._validate_returns(returns)
        self.assets_ = returns.columns.tolist()

        lens_series = self._select_lens(
            returns,
            macros=macros,
            custom_lens=custom_lens,
        ).reindex(self.assets_)

        matrix = returns.to_numpy(dtype=float).T  # assets x time
        scaled = StandardScaler().fit_transform(matrix)

        n_cubes, overlap = self._resolve_cover(regime_value)
        cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)

        eps = self._estimate_epsilon(scaled)
        if not self.epsilon_adaptive:
            # Use the raw quantile as a fixed epsilon when adaptive tuning is disabled.
            eps = max(float(eps), 1e-6)

        clusterer = DBSCAN(
            metric="correlation",
            eps=float(eps),
            min_samples=self.min_cluster_size,
        )

        graph_payload = self.mapper_.map(
            lens_series.to_numpy(dtype=float).reshape(-1, 1),
            scaled,
            cover=cover,
            clusterer=clusterer,
        )
        graph = self._build_graph(graph_payload)
        metrics = self._compute_metrics(graph, lens_series)

        self.cover_params_ = (n_cubes, overlap)
        self.graph_ = graph
        self.metrics_ = metrics
        self._graph_payload = graph_payload
        self.lens_values_ = lens_series
        return self

    def transform(self) -> pd.DataFrame:
        """Return node-level metrics for the fitted Mapper graph."""
        if self.metrics_ is None:
            raise RuntimeError("RegimeAwareMapper must be fitted before calling transform().")
        return self.metrics_.node_metrics.copy()

    def export_graph(self, destination: str | Path) -> tuple[Path, Path]:
        """Persist Mapper graph data to JSON and PNG outputs."""
        if self.graph_ is None or self._graph_payload is None:
            raise RuntimeError("No Mapper graph to export. Call fit() first.")

        dest = Path(destination)
        if dest.suffix:
            json_path = dest.with_suffix(".json")
            png_path = dest.with_suffix(".png")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            json_path = dest / "mapper_graph.json"
            png_path = dest / "mapper_graph.png"

        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self._graph_payload,
            "summary_metrics": self.metrics_.summary if self.metrics_ else {},
            "cover": {
                "n_cubes": self.cover_params_[0],
                "overlap": self.cover_params_[1],
            }
            if self.cover_params_
            else None,
        }
        json_path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")

        self._render_png(self.graph_, png_path)
        return json_path, png_path

    # ------------------------------------------------------------------ #
    # Lens utilities
    def _select_lens(
        self,
        returns: pd.DataFrame,
        *,
        macros: pd.DataFrame | None,
        custom_lens: pd.Series | None,
    ) -> pd.Series:
        if self.lens == "pca_umap":
            return self._lens_pca_umap(returns)
        if self.lens == "volatility":
            return self._lens_volatility(returns)
        if self.lens == "beta_selic":
            return self._lens_macro_beta(returns, macros, "selic")
        if self.lens == "beta_usd":
            return self._lens_macro_beta(returns, macros, "brlusd")
        if self.lens == "beta_ipca":
            return self._lens_macro_beta(returns, macros, "ipca")
        if self.lens == "custom":
            if custom_lens is None:
                raise ValueError("custom_lens must be provided when lens='custom'.")
            return custom_lens.reindex(returns.columns).astype(float)
        raise RuntimeError(f"Unsupported lens '{self.lens}'.")

    def _lens_pca_umap(self, returns: pd.DataFrame) -> pd.Series:
        matrix = returns.to_numpy(dtype=float).T
        n_components = min(matrix.shape[0], matrix.shape[1], 5)
        if n_components < 1:
            raise ValueError("Insufficient observations to compute PCA lens.")
        pca = PCA(n_components=n_components, random_state=self.random_state)
        projected = pca.fit_transform(matrix)
        reducer = UMAP(
            n_components=1,
            random_state=self.random_state,
            n_neighbors=min(15, projected.shape[0] - 1) if projected.shape[0] > 1 else 1,
        )
        embedding = reducer.fit_transform(projected)
        return pd.Series(embedding[:, 0], index=returns.columns, name="lens_pca_umap")

    def _lens_volatility(self, returns: pd.DataFrame) -> pd.Series:
        window = min(len(returns), 252)
        tail = returns.tail(window)
        vol = tail.std(ddof=0).replace(0.0, np.nan)
        if vol.isna().all():
            # Fall back to a Sharpe-like score when volatility is unavailable.
            return self._lens_sharpe(returns)
        vol = vol.fillna(vol.mean()).fillna(0.0)
        return vol.rename("lens_volatility")

    def _lens_sharpe(self, returns: pd.DataFrame) -> pd.Series:
        window = min(len(returns), 252)
        tail = returns.tail(window)
        mean = tail.mean()
        std = tail.std(ddof=0).replace(0.0, np.nan)
        sharpe = (mean * np.sqrt(252)) / std
        sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
        sharpe = sharpe.fillna(sharpe.mean()).fillna(0.0)
        return sharpe.rename("lens_sharpe")

    def _lens_macro_beta(
        self,
        returns: pd.DataFrame,
        macros: pd.DataFrame | None,
        target: str,
    ) -> pd.Series:
        fallback = self._lens_volatility(returns)

        if macros is None or target not in macros:
            logger.warning("Macro series '%s' missing; degrading lens to volatility.", target)
            return fallback

        combined = returns.join(macros[[target]], how="inner").dropna(how="any")
        if len(combined) < 40:
            logger.warning(
                "Insufficient overlap with macro '%s'; degrading lens to volatility.", target
            )
            return fallback

        macro = combined[target].to_numpy(dtype=float)
        macro_centered = macro - np.mean(macro)
        macro_var = float(np.mean(macro_centered**2))
        if np.isclose(macro_var, 0.0):
            logger.warning("Macro variance is zero for '%s'; degrading lens to volatility.", target)
            return fallback

        betas: dict[str, float] = {}
        asset_frame = combined.drop(columns=target)
        for column in asset_frame.columns:
            series = asset_frame[column].to_numpy(dtype=float)
            centered = series - np.mean(series)
            cov = float(np.mean(centered * macro_centered))
            betas[column] = cov / macro_var if macro_var else 0.0

        beta_series = pd.Series(betas, name=f"lens_beta_{target}")
        if beta_series.isna().all():
            return fallback
        return beta_series.fillna(beta_series.mean()).rename(f"lens_beta_{target}")

    # ------------------------------------------------------------------ #
    # Helpers
    def _validate_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas.DataFrame.")
        if returns.empty:
            raise ValueError("returns must contain observations.")
        cleaned = returns.dropna(axis=1, how="all").copy()
        if cleaned.empty:
            raise ValueError("returns requires at least one asset with data.")
        cleaned = cleaned.fillna(0.0)
        return cleaned

    def _resolve_cover(self, regime_value: float | None) -> tuple[int, float]:
        n_cubes = self.base_n_cubes
        overlap = self.base_overlap
        if regime_value is not None:
            rv = float(np.clip(regime_value, 0.0, 1.0))
            n_cubes = max(1, int(round(self.base_n_cubes * (1.0 + 0.5 * rv))))
            overlap = float(np.clip(self.base_overlap + 0.2 * rv, 0.05, 0.95))
        return n_cubes, overlap

    def _estimate_epsilon(self, matrix: np.ndarray) -> float:
        if matrix.shape[0] < 2:
            return 0.5
        distances = pairwise_distances(matrix, metric="correlation")
        upper = distances[np.triu_indices_from(distances, k=1)]
        finite = upper[np.isfinite(upper)]
        if finite.size == 0:
            return 0.5
        quantile = float(np.quantile(finite, self.eps_quantile))
        if not self.epsilon_adaptive:
            return max(quantile, 1e-6)
        local_density = float(np.mean(finite))
        if local_density < quantile:
            quantile *= 0.9
        return max(quantile, 1e-6)

    def _build_graph(self, payload: Mapping[str, Any]) -> nx.Graph:
        graph = nx.Graph()
        nodes = payload.get("nodes", {})
        for key, members in nodes.items():
            node_id = str(key)
            graph.add_node(node_id, members=members, size=len(members))

        for key, neighbors in payload.get("links", {}).items():
            node_id = str(key)
            for neighbor in neighbors:
                graph.add_edge(node_id, str(neighbor))
        return graph

    def _compute_metrics(self, graph: nx.Graph, lens_series: pd.Series) -> MapperMetrics:
        if graph.number_of_nodes() == 0:
            empty = pd.DataFrame(
                columns=[
                    "size",
                    "members",
                    "degree_centrality",
                    "eigenvector_centrality",
                    "betweenness_centrality",
                    "peripherality_score",
                    "lens_mean",
                ]
            )
            summary = {"n_components": 0.0, "avg_degree": 0.0, "gini_node_size": 0.0}
            return MapperMetrics(summary=summary, node_metrics=empty, graph=graph)

        sizes = np.array([data.get("size", 0) for _, data in graph.nodes(data=True)], dtype=float)
        degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
        avg_degree = float(degrees.mean()) if degrees.size else 0.0
        gini = _gini(sizes)
        n_components = float(nx.number_connected_components(graph))

        degree_cent = nx.degree_centrality(graph)
        try:
            eigen_cent = nx.eigenvector_centrality(graph, max_iter=2000)
        except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
            eigen_cent = {node: np.nan for node in graph.nodes()}
        betweenness = nx.betweenness_centrality(graph, normalized=True)

        assets = self.assets_ or []
        records = []
        for node, data in graph.nodes(data=True):
            members_idx = data.get("members", [])
            members = [assets[i] for i in members_idx if i < len(assets)]
            members_series = lens_series.reindex(members) if members else None
            lens_mean = float(members_series.mean()) if members_series is not None else np.nan
            deg_cent = degree_cent.get(node, 0.0)

            records.append(
                {
                    "node": node,
                    "size": data.get("size", 0),
                    "members": members,
                    "degree_centrality": deg_cent,
                    "eigenvector_centrality": eigen_cent.get(node, np.nan),
                    "betweenness_centrality": betweenness.get(node, 0.0),
                    "peripherality_score": 1.0 - deg_cent,
                    "lens_mean": lens_mean,
                }
            )

        node_metrics = pd.DataFrame.from_records(records).set_index("node")
        summary = {
            "n_components": n_components,
            "avg_degree": avg_degree,
            "gini_node_size": gini,
        }
        return MapperMetrics(summary=summary, node_metrics=node_metrics, graph=graph)

    def _render_png(self, graph: nx.Graph, output: Path) -> None:
        plt.figure(figsize=(6, 4))
        layout = nx.spring_layout(graph, seed=self.random_state)
        node_sizes = [
            max(120.0, data.get("size", 1) * 90.0)
            for _, data in graph.nodes(data=True)
        ]
        nx.draw_networkx(
            graph,
            pos=layout,
            with_labels=True,
            node_color="cornflowerblue",
            node_size=node_sizes,
            edge_color="gray",
            font_size=8,
        )
        plt.axis("off")
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output, dpi=200)
        plt.close()


def _gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0 or np.allclose(arr.sum(), 0.0):
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(index * sorted_arr)) / (n * sorted_arr.sum()) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable.")
