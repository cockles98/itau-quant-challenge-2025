from __future__ import annotations

import logging
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

__all__ = ["takens_embedding", "mapper_graph", "tfi_score", "TFIParams"]


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

        return cls(
            delay=int(pick("delay", 1)),
            dim=int(pick("dim", 3)),
            n_cubes=int(pick("n_cubes", 8)),
            overlap=float(pick("overlap", 0.75)),
            epsilon=_eps,
            min_samples=int(pick("min_samples", 2)),
            window=int(_win),
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
            if pts.shape[0] >= 2:
                d = pairwise_distances(pts)
                # remove zeros/NaNs e toma um quantil baixo das distâncias
                d = d[np.isfinite(d)]
                d = d[d > 0]
                if d.size:
                    eps = float(np.nanpercentile(d, 10.0))
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
    if prices.isnull().any().any():
        prices = prices.ffill().dropna()
    # tudo numérico
    prices = prices.apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")

    scores: dict[pd.Timestamp, float] = {}

    for end_idx in range(params.window, len(prices) + 1):
        window_prices = prices.iloc[end_idx - params.window : end_idx]
        lr = np.log(window_prices).diff().dropna()

        per_asset_scalar: list[float] = []
        for col in lr.columns:
            arr = lr[col].values
            # precisa de pontos suficientes pro embedding
            if len(arr) < (params.dim - 1) * params.delay + 1:
                continue
            try:
                emb = takens_embedding(arr, delay=params.delay, dim=params.dim)
                g = mapper_graph(
                    emb,
                    n_cubes=params.n_cubes,
                    overlap=params.overlap,
                    epsilon=params.epsilon,
                    min_samples=params.min_samples,
                    scale=True,  # sempre escalar o embedding
                )
                feats = _graph_features(g)
                # ESCALAR escolhido para o TFI local:
                per_asset_scalar.append(float(feats["edge_density"]))
            except Exception as exc:  # não derrubar o loop por série problemática
                logger.debug("Skipping series %s due to error: %s", col, exc)
                continue

        local = float(np.nanmean(per_asset_scalar)) if per_asset_scalar else 0.0
        scores[window_prices.index[-1]] = local

    ser = pd.Series(scores, name="tfi_score").astype(float)
    
    # ===== TESTE DIAGNÓSTICO =====
    import os
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
    return ser
