from __future__ import annotations

import logging
from collections import OrderedDict
import hashlib
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


logger = logging.getLogger(__name__)


_TFI_CACHE: OrderedDict[tuple[str, TFIParams], pd.Series] = OrderedDict()
_TFI_CACHE_MAXSIZE = 8


def _cache_lookup(key: tuple[str, TFIParams]) -> pd.Series | None:
    try:
        value = _TFI_CACHE.pop(key)
    except KeyError:
        return None
    _TFI_CACHE[key] = value
    return value.copy()


def _cache_store(key: tuple[str, TFIParams], value: pd.Series) -> None:
    if len(_TFI_CACHE) >= _TFI_CACHE_MAXSIZE:
        _TFI_CACHE.popitem(last=False)
    _TFI_CACHE[key] = value.copy()


def _frame_signature(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=False)
    digest = hashlib.blake2b(hashed.values.tobytes(), digest_size=16).hexdigest()
    return digest
__all__ = ["takens_embedding", "mapper_graph", "mapper_for_asset", "tfi_score", "TFIParams"]


# ----------------------------
# Hiperparâmetros do TFI
# ----------------------------
@dataclass(frozen=True)
class TFIParams:
    delay: int = 1
    dim: int = 2
    n_cubes: int = 6
    overlap: float = 0.5
    epsilon: Optional[float] = None  # None => usa epsilon adaptativo no mapper
    min_samples: int = 2
    window: int = 126
    eps_quantile: Optional[float] = None
    smooth_span: Optional[int] = None
    # ---------- NOVO: fábrica robusta a diferentes layouts de YAML ----------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any], vol_window_fallback: int = 126) -> "TFIParams":
        """
        Constrói TFIParams a partir do dict de config, aceitando múltiplos namespaces:
        - cfg["tda"].*
        - cfg["features"]["tfi"].*
        - topo do YAML (quando o tuner injeta delay/dim/n_cubes/overlap diretamente)
        Também harmoniza epsilon=None/"none" e define window sensível a windows.vol_window.
        """
        def _ns(d: Dict[str, Any], path: Iterable[str]) -> Dict[str, Any]:
            cur: Any = d
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return {}
            return cur if isinstance(cur, dict) else {}

        # Candidatos em ordem de prioridade
        ns_list = [
            _ns(cfg, ("tda",)),
            _ns(cfg, ("features", "tfi")),
            cfg if isinstance(cfg, dict) else {},
        ]

        def pick(key: str, default: Any) -> Any:
            for ns in ns_list:
                if key in ns:
                    return ns[key]
            return default

        # epsilon pode vir como None/"none"/"null"
        _eps_raw = pick("epsilon", None)
        if _eps_raw is None or str(_eps_raw).lower() in {"none", "null"}:
            _eps = None
        else:
            _eps = float(_eps_raw)

        # window: respeita uma chave explícita, senão tenta windows.vol_window, senão fallback
        _win_raw = pick("window", None)
        if _win_raw is None:
            try:
                vw = int(_ns(cfg, ("windows",)).get("vol_window", vol_window_fallback))
            except Exception:
                vw = int(vol_window_fallback)
            _win = max(int(vw), 63)
        else:
            _win = int(_win_raw)

        _eps_quantile_raw = pick("eps_quantile", None)
        try:
            _eps_quantile = None if _eps_quantile_raw is None else float(_eps_quantile_raw)
        except (TypeError, ValueError):
            _eps_quantile = None
        if _eps_quantile is None:
            _eps_quantile = 0.1

        _smooth_raw = pick("smooth_span", None)
        try:
            _smooth_span = None if _smooth_raw is None else int(_smooth_raw)
        except (TypeError, ValueError):
            _smooth_span = None

        return cls(
            delay=int(pick("delay", 1)),
            dim=int(pick("dim", 3)),
            n_cubes=int(pick("n_cubes", 8)),
            overlap=float(pick("overlap", 0.75)),
            epsilon=_eps,
            min_samples=int(pick("min_samples", 2)),
            window=int(_win),
            eps_quantile=float(_eps_quantile),
            smooth_span=_smooth_span,
        )

# ----------------------------
# Takens embedding
# ----------------------------
def takens_embedding(series: Iterable[float], delay: int, dim: int) -> np.ndarray:
    """Constrói o embedding de Takens de uma série univariada.

    series: array-like 1D (float)
    delay: atraso entre coordenadas (>0)
    dim: dimensão do embedding (>=2)
    """
    data = np.asarray(series, dtype=float)
    if delay <= 0:
        raise ValueError("delay must be positive")
    if dim <= 1:
        raise ValueError("dim must be at least 2")
    need = (dim - 1) * delay + 1
    if len(data) < need:
        raise ValueError(f"series length insufficient for embedding: need >= {need}")

    n_vectors = len(data) - (dim - 1) * delay
    # colunas: [x_t, x_{t+delay}, ..., x_{t+(dim-1)delay}]
    return np.column_stack([data[i : i + n_vectors] for i in range(0, dim * delay, delay)])


# ----------------------------
# Mapper-like graph
# ----------------------------
def mapper_graph(
    embedded: np.ndarray,
    n_cubes: int,
    overlap: float,
    metric: str = "euclidean",
    epsilon: Optional[float] = None,
    min_samples: int = 3,
    scale: bool = True,
    eps_quantile: float = 0.10,
) -> nx.Graph:
    """Constrói um grafo ao estilo Mapper com binning grosso + DBSCAN por intervalo.

    - Aplica StandardScaler em `embedded` quando `scale=True`.
    - Define a lente como a 1ª coordenada do embedding escalado (simples e robusto).
    - Cria intervalos sobrepostos ao longo da lente.
    - Clusteriza pontos de cada intervalo com DBSCAN.
    - Cria arestas entre nós de cubos **iguais ou adjacentes** quando há interseção
      não vazia nos conjuntos de membros (pontos compartilhados).
    """
    if embedded.ndim != 2:
        raise ValueError("embedded must be 2D array")
    if embedded.shape[0] < 2:
        raise ValueError("embedded must contain at least two samples")
    if n_cubes <= 0:
        raise ValueError("n_cubes must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    # 1) Scaling
    X = StandardScaler().fit_transform(embedded) if scale else embedded

    # 2) Lente simples: 1ª coordenada (se quiser PCA, troque por PCA(n_components=1))
    lens = X[:, 0]
    min_val, max_val = lens.min(), lens.max()
    if max_val == min_val:
        max_val += 1e-12

    cube_size = (max_val - min_val) / float(n_cubes)
    overlap_th = cube_size * float(overlap)

    # 3) Intervalos sobrepostos + indices globais dos pontos
    intervals: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cube in range(n_cubes):
        left = min_val + cube * cube_size - overlap_th
        right = min_val + (cube + 1) * cube_size + overlap_th
        mask = (lens >= left) & (lens <= right)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        intervals.append((cube, idx, X[idx]))

    # 4) DBSCAN por intervalo (guardando membros)
    g = nx.Graph()
    cluster_id = 0
    for cube_idx, idx_in_cube, pts in intervals:
        # eps = float(epsilon) if epsilon is not None else (cube_size / 2.0)
        # labels = DBSCAN(eps=eps, min_samples=int(min_samples), metric=metric).fit_predict(pts)
        # Epsilon adaptativo por cubo: evita regime “sem cluster” (grafo vazio)
        if epsilon is None:
            quant = float(np.clip(eps_quantile, 1e-6, 0.5))
            if pts.shape[0] >= 2:
                d = pairwise_distances(pts)
                # remove zeros/NaNs e toma um quantil baixo das distâncias
                d = d[np.isfinite(d)]
                d = d[d > 0]
                if d.size:
                    eps = float(np.nanpercentile(d, quant * 100.0))
                else:
                    eps = 0.5
            else:
                eps = 0.5
        else:
            eps = float(epsilon)
        labels = DBSCAN(eps=eps, min_samples=int(min_samples), metric=metric)\
            .fit_predict(pts)
        for lbl in set(labels):
            if lbl == -1:
                continue
            members = idx_in_cube[labels == lbl]
            if members.size == 0:
                continue
            node = f"{cube_idx}_{cluster_id}"
            g.add_node(
                node,
                cube=int(cube_idx),
                size=int(members.size),
                members=set(map(int, members)),
            )
            cluster_id += 1

    # 5) Arestas por interseção entre cubos (mesmo ou adjacentes)
    nodes = list(g.nodes)
    for i in range(len(nodes)):
        ci = g.nodes[nodes[i]]["cube"]
        mi = g.nodes[nodes[i]]["members"]
        for j in range(i + 1, len(nodes)):
            cj = g.nodes[nodes[j]]["cube"]
            if abs(ci - cj) <= 1:  # mesmo cubo ou adjacente
                mj = g.nodes[nodes[j]]["members"]
                if mi & mj:  # interseção não vazia
                    g.add_edge(nodes[i], nodes[j])

    return g


# ----------------------------
# Features do grafo
# ----------------------------
def _graph_features(graph: nx.Graph) -> Dict[str, float]:
    """Extrai métricas simples de conectividade (inclusive densidade)."""
    n = graph.number_of_nodes()
    if n == 0:
        return {
            "avg_degree": 0.0,
            "avg_degree_norm": 0.0,
            "clustering": 0.0,
            "components": 0.0,
            "edge_density": 0.0,
        }
    m = graph.number_of_edges()
    degrees = [deg for _, deg in graph.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    avg_degree_norm = (avg_degree / (n - 1)) if n > 1 else 0.0
    clustering = float(nx.average_clustering(graph)) if n > 1 else 0.0
    components = float(nx.number_connected_components(graph))
    edge_density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0

    return {
        "avg_degree": avg_degree,
        "avg_degree_norm": float(avg_degree_norm),
        "clustering": clustering,
        "components": components,
        "edge_density": float(edge_density),
    }



def mapper_for_asset(
    prices: pd.DataFrame,
    params: TFIParams,
    asset: str,
    *,
    end_date: pd.Timestamp | None = None,
    return_embedding: bool = False,
) -> tuple[nx.Graph, dict[str, float]] | tuple[nx.Graph, np.ndarray, dict[str, float]]:
    """Gera o grafo Mapper para um ativo específico e período final.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame de preços com datas no índice e ativos nas colunas.
    params : TFIParams
        Hiperparâmetros do embedding e do Mapper.
    asset : str
        Ativo alvo (deve ser coluna de ``prices``).
    end_date : pd.Timestamp | None, opcional
        Última data a considerar (inclusive). Se ``None``, usa a última disponível.
    return_embedding : bool, opcional
        Se ``True`` retorna também a matriz de embedding de Takens.

    Returns
    -------
    tuple
        ``(graph, metadata)`` ou ``(graph, embedding, metadata)`` quando
        ``return_embedding=True``. O metadata inclui métricas do grafo e
        informações da janela.
    """
    if params is None:
        params = TFIParams()

    data = prices.sort_index()
    if asset not in data.columns:
        raise KeyError(f"Asset '{asset}' not found in price data")

    if end_date is not None:
        data = data.loc[: end_date]
        if data.empty:
            raise ValueError("No observations available up to end_date")

    series = data[asset].dropna()
    window = int(params.window)
    if len(series) < window:
        raise ValueError(
            f"Insufficient history for asset '{asset}' (need >= {window} observations)"
        )

    window_series = series.iloc[-window:]
    log_prices = np.log(window_series.to_numpy(dtype=float))
    if not np.isfinite(log_prices).all():
        raise ValueError("Price series contains non-finite values after log transform")

    returns = np.diff(log_prices)
    need = (params.dim - 1) * params.delay + 1
    if len(returns) < need:
        raise ValueError(
            "Not enough returns to build Takens embedding with current parameters"
        )

    embedding = takens_embedding(returns, delay=params.delay, dim=params.dim)
    graph = mapper_graph(
        embedding,
        n_cubes=params.n_cubes,
        overlap=params.overlap,
        epsilon=params.epsilon,
        min_samples=params.min_samples,
        scale=True,
        eps_quantile=float(params.eps_quantile) if params.eps_quantile is not None else 0.1,
    )

    metrics = _graph_features(graph)
    window_start = pd.Timestamp(window_series.index[0])
    window_end = pd.Timestamp(window_series.index[-1])
    metadata = {
        "asset": asset,
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "length": window,
        },
        "params": {
            "delay": params.delay,
            "dim": params.dim,
            "n_cubes": params.n_cubes,
            "overlap": params.overlap,
            "epsilon": params.epsilon,
            "min_samples": params.min_samples,
            "eps_quantile": float(params.eps_quantile) if params.eps_quantile is not None else 0.1,
            "smooth_span": params.smooth_span,
        },
        "graph_metrics": metrics,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }

    if return_embedding:
        return graph, embedding, metadata
    return graph, metadata

# ----------------------------
# TFI score (regime)
# ----------------------------
def tfi_score(prices: pd.DataFrame, params: TFIParams | None = None) -> pd.Series:
    """Calcula um escalar de regime por data a partir da conectividade do grafo.

    - Para cada janela: monta o grafo por ativo, extrai um ESCALAR de conectividade
      (por padrão, `edge_density`) e tira a média cross-sectional entre ativos.
    - Ao final, normaliza **ao longo do tempo** (min–max) para [0,1].
    """
    if params is None:
        params = TFIParams()

    # sanity nos dados
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.ffill()
    prices = prices.dropna(axis=1, how="all").dropna(how="all")
    # tudo numerico

    use_cache = os.getenv("TFI_CACHE_DISABLE", "0") != "1"
    signature = _frame_signature(prices)
    cache_key = (signature, params)
    if use_cache:
        cached = _cache_lookup(cache_key)
        if cached is not None:
            logger.debug("TFI cache hit for signature %s", signature)
            return cached
    scores: dict[pd.Timestamp, float] = {}

    for end_idx in range(params.window, len(prices) + 1):
        window_prices = prices.iloc[end_idx - params.window : end_idx]
        per_asset_scalar: list[float] = []
        need = (params.dim - 1) * params.delay + 1
        for col in window_prices.columns:
            series = window_prices[col].dropna()
            if len(series) < params.window:
                continue
            values = series.to_numpy(dtype=float)
            window_values = values[-params.window :]
            returns = np.diff(np.log(window_values))
            if len(returns) < need:
                continue
            if not np.isfinite(returns).all():
                continue
            try:
                emb = takens_embedding(returns, delay=params.delay, dim=params.dim)
                g = mapper_graph(
                    emb,
                    n_cubes=params.n_cubes,
                    overlap=params.overlap,
                    epsilon=params.epsilon,
                    min_samples=params.min_samples,
                    scale=True,  # sempre escalar o embedding
                    eps_quantile=float(params.eps_quantile) if params.eps_quantile is not None else 0.1,
                )
                feats = _graph_features(g)
                per_asset_scalar.append(float(feats["edge_density"]))
            except Exception as exc:  # nao derrubar o loop por serie problematica
                logger.debug("Skipping series %s due to error: %s", col, exc)
                continue
        local = float(np.nanmean(per_asset_scalar)) if per_asset_scalar else 0.0
        scores[window_prices.index[-1]] = local

    ser = pd.Series(scores, name="tfi_score").astype(float)
    
    # ===== TESTE DIAGNÓSTICO =====
    # Se REGIME "fake" estiver ligado, injeta um seno para checar sensibilidade do pipeline.
    if os.getenv("TFI_FAKE_SINE", "0") == "1":
        t = np.arange(len(ser), dtype=float)
        ser = pd.Series(0.5 + 0.5 * np.sin(2*np.pi*t/63.0), index=ser.index, name="tfi_score")
    # ===== FIM TESTE =====

    if ser.empty:
        return ser

    # # Normalização AO LONGO DO TEMPO (agora sim TFI varia!)
    # smin, smax = float(ser.min()), float(ser.max())
    # if smax > smin:
    #     ser = (ser - smin) / (smax - smin)
    # else:
    #     ser = ser * 0.0
    # ser = ser.clip(0.0, 1.0)
    # Normalização robusta ao longo do tempo (evita colapso para 0/constante)
    q1, q9 = np.nanpercentile(ser.values, [5, 95])
    den = max(1e-9, float(q9 - q1))
    ser = ((ser - q1) / den).clip(0.0, 1.0)
    if params.smooth_span and params.smooth_span > 1:
        ser = ser.ewm(span=int(params.smooth_span), adjust=False).mean()
    if use_cache:
        _cache_store(cache_key, ser)
    return ser

