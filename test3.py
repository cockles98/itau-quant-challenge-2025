# tests.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# importa nossas funções do projeto
from src.features.tda import *

ARTIF_DIR = Path("./artifacts/tests")
ARTIF_DIR.mkdir(parents=True, exist_ok=True)


def load_prices(path: Optional[str]) -> pd.DataFrame:
    """
    Lê um CSV de preços no formato:
      index = datas, columns = tickers/ativos, valores = preços
    Se não for informado, gera uma série sintética com 2 regimes (quebra estrutural),
    suficiente para o DBSCAN enxergar densidades diferentes.
    """
    if path and Path(path).exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # garante numeric
        df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")
        return df

    # --- série sintética fallback ---
    n = 600
    rng = np.random.default_rng(42)
    # regime 1: baixa vol
    r1 = rng.normal(0.0004, 0.005, size=n // 2)
    # regime 2: alta vol
    r2 = rng.normal(0.0001, 0.02, size=n - n // 2)
    rets = np.r_[r1, r2]
    prices = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"SYN": prices}, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    return df


def _embed_from_prices(prices: pd.DataFrame, delay: int, dim: int) -> np.ndarray:
    """
    Cria um embedding univariado sobre o 1º ativo (suficiente para diagnóstico).
    Usa preços -> retornos log -> normaliza nível para evitar escala explosiva.
    """
    ser = np.log(prices.iloc[:, 0]).diff().dropna().to_numpy()
    return takens_embedding(ser, delay=delay, dim=dim)


def diagnose_mapper(
    embedded: np.ndarray,
    n_cubes: int,
    overlap: float,
    eps_grid: Iterable[float],
    min_samples_grid: Iterable[int],
    scale: bool,
    tag: str,
) -> pd.DataFrame:
    """
    Roda o mapper_graph em um grid e retorna um DataFrame com (#nodes, #components).
    """
    if scale:
        scaler = StandardScaler()
        emb = scaler.fit_transform(embedded)
    else:
        emb = embedded

    rows = []
    for eps in eps_grid:
        for ms in min_samples_grid:
            g = mapper_graph(
                emb,
                n_cubes=n_cubes,
                overlap=overlap,
                epsilon=float(eps),
                min_samples=int(ms),
            )
            rows.append(
                {
                    "epsilon": float(eps),
                    "min_samples": int(ms),
                    "nodes": g.number_of_nodes(),
                    "components": (0 if g.number_of_nodes() == 0 else int(nx_number_cc(g))),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(ARTIF_DIR / f"mapper_diag_{tag}.csv", index=False)
    print(f"[diag] mapper_diag_{tag}.csv salvo ({len(out)} linhas).")
    return out


def nx_number_cc(g):
    # helper (evita importar networkx aqui no topo)
    import networkx as nx
    return nx.number_connected_components(g)


def nx_avg_clustering(g):
    import networkx as nx
    if g.number_of_nodes() <= 1:
        return 0.0
    return nx.average_clustering(g)


def features_by_window(
    prices: pd.DataFrame, params: TFIParams, sample_stride: int = 5
) -> pd.DataFrame:
    """
    Replica o miolo do tfi_score: para cada janela, construímos o grafo e
    extraímos as features topológicas brutas. Amostramos 1 a cada `sample_stride`
    janelas para reduzir custo.
    """
    feats = []
    # forward-fill e dropna como no tfi_score
    px = prices.ffill().dropna()
    for end_idx in range(params.window, len(px) + 1, sample_stride):
        window_prices = px.iloc[end_idx - params.window : end_idx]
        ser = np.log(window_prices.iloc[:, 0]).diff().dropna().to_numpy()

        if len(ser) < params.dim * params.delay + 2:
            continue

        embedded = takens_embedding(ser, delay=params.delay, dim=params.dim)

        graph = mapper_graph(
            embedded,
            n_cubes=params.n_cubes,
            overlap=params.overlap,
            epsilon=params.epsilon,
            min_samples=params.min_samples,
        )

        # mesmas métricas que _graph_features em tda.py
        nodes = graph.number_of_nodes()
        if nodes == 0:
            avg_degree = 0.0
            clustering = 0.0
            components = 0
        else:
            import networkx as nx
            avg_degree = float(np.mean([deg for _, deg in graph.degree()]))
            clustering = float(nx_avg_clustering(graph))
            components = int(nx.number_connected_components(graph))

        feats.append(
            {
                "date": window_prices.index[-1],
                "nodes": nodes,
                "components": components,
                "avg_degree": avg_degree,
                "clustering": clustering,
            }
        )
    out = pd.DataFrame(feats).set_index("date")
    out.to_csv(ARTIF_DIR / "features_by_window.csv")
    print(f"[diag] features_by_window.csv salvo ({len(out)} janelas).")
    return out


def sweep_tfi(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Varre combinações de TFIParams e mede:
      - % de zeros
      - nº de valores únicos
      - min/max/média do score
    """
    grid = []
    for eps in [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        for ms in [2, 3, 4]:
            for n_cubes in [4, 6, 8, 12]:
                for ov in [0.25, 0.5, 0.75]:
                    params = TFIParams(
                        delay=1,
                        dim=3,
                        window=126,
                        epsilon=eps,
                        min_samples=ms,
                        n_cubes=n_cubes,
                        overlap=ov,
                    )

                    s = tfi_score(prices.copy(), params=params)
                    vals = s.values
                    uniq = len(np.unique(np.round(vals, 6)))
                    pct_zeros = float(np.mean(vals == 0.0)) if len(vals) else 0.0
                    grid.append(
                        {
                            "epsilon": eps,
                            "min_samples": ms,
                            "n_cubes": n_cubes,
                            "overlap": ov,
                            "len": int(len(vals)),
                            "pct_zeros": pct_zeros,
                            "n_unique": uniq,
                            "min": float(np.min(vals)) if len(vals) else np.nan,
                            "max": float(np.max(vals)) if len(vals) else np.nan,
                            "mean": float(np.mean(vals)) if len(vals) else np.nan,
                        }
                    )
    out = pd.DataFrame(grid)
    out.to_csv(ARTIF_DIR / "tfi_sweep.csv", index=False)
    print(f"[diag] tfi_sweep.csv salvo ({len(out)} linhas).")
    return out


def sanity_check_synthetic():
    """
    Gera um dataset com 2 nuvens gaussianas em 2D (regimes distintos), projeta num embedding simples
    e mede se surgem nós/componente > 0 com eps/min_samples razoáveis.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0, 0], scale=[0.5, 0.5], size=(400, 2))
    b = rng.normal(loc=[3, 0], scale=[0.4, 0.6], size=(400, 2))
    X = np.vstack([a, b])
    # “embedding” 2D artificial só para passar pelo mapper
    g = mapper_graph(X, n_cubes=6, overlap=0.5, epsilon=0.25, min_samples=2)
    import networkx as nx
    return {
        "nodes": g.number_of_nodes(),
        "components": int(nx.number_connected_components(g)) if g.number_of_nodes() else 0,
    }


# --- NOVOS DIAGNÓSTICOS DE ARESTAS ---

def graph_edge_stats(embedded, n_cubes, overlap, eps, min_samples, scale=False):
    from sklearn.preprocessing import StandardScaler
    emb = StandardScaler().fit_transform(embedded) if scale else embedded
    g = mapper_graph(
        emb, n_cubes=n_cubes, overlap=overlap,
        epsilon=eps, min_samples=min_samples
    )
    import networkx as nx
    n = g.number_of_nodes()
    m = g.number_of_edges()
    iso = int(sum(1 for _, d in g.degree() if d == 0))
    comps = int(nx.number_connected_components(g)) if n else 0
    avg_deg = (0.0 if n == 0 else float(np.mean([d for _, d in g.degree()])))
    return {
        "nodes": n, "edges": m, "isolated": iso, "components": comps,
        "avg_degree": avg_deg, "pct_isolated": (iso / n if n else 0.0)
    }

def sweep_edges(embedded):
    rows = []
    for scale in [False, True]:
        for ov in [0.25, 0.5, 0.75]:
            for eps in [0.02, 0.05, 0.10, 0.20]:
                for ms in [2, 3, 4]:
                    s = graph_edge_stats(
                        embedded, n_cubes=8, overlap=ov,
                        eps=eps, min_samples=ms, scale=scale
                    )
                    s.update({"scale": "zscore" if scale else "raw",
                              "overlap": ov, "epsilon": eps, "min_samples": ms})
                    rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(ARTIF_DIR / "edge_sweep.csv", index=False)
    print(f"[diag] edge_sweep.csv salvo ({len(df)} linhas).")
    print(df.groupby(["scale","overlap"]).agg(
        nodes_mean=("nodes","mean"),
        edges_mean=("edges","mean"),
        pct_iso_mean=("pct_isolated","mean")
    ).round(3))
    return df

def features_by_window_with_edges(prices: pd.DataFrame, params: TFIParams, sample_stride: int = 5):
    feats = []
    px = prices.ffill().dropna()
    for end_idx in range(params.window, len(px) + 1, sample_stride):
        window_prices = px.iloc[end_idx - params.window : end_idx]
        ser = np.log(window_prices.iloc[:, 0]).diff().dropna().to_numpy()
        if len(ser) < params.dim * params.delay + 2:
            continue
        embedded = takens_embedding(ser, delay=params.delay, dim=params.dim)
        g = mapper_graph(
            embedded, n_cubes=params.n_cubes, overlap=params.overlap,
            epsilon=params.epsilon, min_samples=params.min_samples
        )
        import networkx as nx
        n = g.number_of_nodes()
        m = g.number_of_edges()
        iso = int(sum(1 for _, d in g.degree() if d == 0))
        comps = int(nx.number_connected_components(g)) if n else 0
        avg_deg = 0.0 if n == 0 else float(np.mean([d for _, d in g.degree()]))
        clust = 0.0 if n <= 1 else float(nx.average_clustering(g))
        feats.append({
            "date": window_prices.index[-1],
            "nodes": n, "edges": m, "isolated": iso, "pct_isolated": (iso/n if n else 0.0),
            "components": comps, "avg_degree": avg_deg, "clustering": clust
        })
    out = pd.DataFrame(feats).set_index("date")
    out.to_csv(ARTIF_DIR / "features_with_edges_by_window.csv")
    print(f"[diag] features_with_edges_by_window.csv salvo ({len(out)} janelas).")
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnósticos para TFI/Mapper")
    ap.add_argument("--prices", type=str, default=None,
                    help="CSV de preços (index=data, columns=ativos). Se ausente, usa série sintética.")
    args = ap.parse_args()

    prices = load_prices(args.prices)
    print(f"[info] prices shape={prices.shape}, cols={list(prices.columns)[:5]}")

    # ----- 2) FEATURES por janela -----
    params = TFIParams(
    delay=1,
    dim=3,
    window=126,
    n_cubes=8,
    overlap=0.75,
    epsilon=0.40,
    min_samples=2,
    )

    # ----- 1) EMBEDDING e MAPPER (com e sem scaling) -----
    emb = _embed_from_prices(prices, delay=1, dim=3)
    print(f"[info] embedded shape={emb.shape}")

    # NOVO: varredura de arestas
    _ = sweep_edges(emb)

    # NOVO: features por janela incluindo arestas
    feats_edges = features_by_window_with_edges(prices, params=params, sample_stride=5)
    if len(feats_edges):
        print(feats_edges[["nodes","edges","pct_isolated","components","avg_degree","clustering"]]
            .describe().round(3))


    eps_grid = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00]
    min_samples_grid = [2, 3, 4]

    diag_noscale = diagnose_mapper(
        emb, n_cubes=8, overlap=0.5, eps_grid=eps_grid,
        min_samples_grid=min_samples_grid, scale=False, tag="noscale"
    )
    diag_scale = diagnose_mapper(
        emb, n_cubes=8, overlap=0.5, eps_grid=eps_grid,
        min_samples_grid=min_samples_grid, scale=True, tag="zscore"
    )

    # rápido resumo no terminal
    def quick_report(df: pd.DataFrame, label: str):
        any_nodes = (df["nodes"] > 0).any()
        any_comp = (df["components"] > 0).any()
        print(f"[{label}] algum nó? {any_nodes} | algum componente? {any_comp}")
        if not any_nodes:
            print(f"[{label}] (sinal forte de eps/min_samples inadequados ou escala ruim)")

    quick_report(diag_noscale, "noscale")
    quick_report(diag_scale, "zscore")


    feats = features_by_window(prices, params=params, sample_stride=5)
    if len(feats):
        print("[features] percentuais de zeros:")
        for col in ["nodes", "components", "avg_degree", "clustering"]:
            print(f"  {col}: {float((feats[col] == 0).mean()):.3f}")

    # ----- 3) Varredura do TFI -----
    sweep = sweep_tfi(prices)
    print(sweep.sort_values(["pct_zeros", "n_unique"]).head(10).to_string(index=False))

    # ----- 4) Sanidade com dado sintético -----
    san = sanity_check_synthetic()
    print(f"[sanity] synthetic -> nodes={san['nodes']}, components={san['components']} (esperado > 0)")

    print(f"\nArquivos gerados em: {ARTIF_DIR.resolve()}")

if __name__ == "__main__":
    main()
