## Parte 1
#import pandas as pd
#from pathlib import Path
#
#project_root = Path.cwd().resolve().parent  # inside notebooks/
#CSV = project_root / "itau-quant-challenge-2025" / "data" / "ibov_b3_2019_2022.csv"   # ajuste o caminho se preciso
#print(CSV)
#df = pd.read_csv(CSV)
#df.columns = [c.lower() for c in df.columns]
#df["date"] = pd.to_datetime(df["date"])
#
## Preços em formato wide: índice=Data, colunas=Tickers
#prices = (df.pivot(index="date", columns="asset", values="close")
#            .sort_index()
#            .ffill()           # preenche buracos pequenos
#            .dropna(how="all"))
#
## Parte 2
#import importlib, sys
#sys.path.append("src/features")         # ou a pasta onde está o tda.py
#tda = importlib.import_module("tda")
#
#TFIParams = tda.TFIParams
#tfi_score = tda.tfi_score
#
## Grid enxuto — pode ampliar depois
#grid = [
#    {"epsilon": 0.10, "n_cubes": 5},
#    {"epsilon": 0.30, "n_cubes": 5},
#    {"epsilon": 0.60, "n_cubes": 5},
#    {"epsilon": 0.10, "n_cubes": 10},
#    {"epsilon": 0.30, "n_cubes": 10},
#    {"epsilon": 0.60, "n_cubes": 10},
#]
#
#params_base = dict(delay=1, dim=3, overlap=1.0, min_samples=3, window=63)
#
#tfi_dict = {}
#for g in grid:
#    p = TFIParams(n_cubes=g["n_cubes"], epsilon=g["epsilon"], **params_base)
#    ser = tfi_score(prices, params=p)     # <- Série (índice=Data), valores em [0,1]
#    ser.name = f"eps={g['epsilon']}, cubes={g['n_cubes']}"
#    tfi_dict[ser.name] = ser
#
#tfi_df = pd.concat(tfi_dict.values(), axis=1).dropna(how="all")
#
## Parte 3
#summary = pd.DataFrame({
#    "min":  tfi_df.min(),
#    "max":  tfi_df.max(),
#    "mean": tfi_df.mean(),
#    "std":  tfi_df.std(ddof=0),
#}).round(4)
#
#print(summary)
## (Opcional) correlação entre as séries de TFI, para ver se mudam pouco ou nada:
#print("\nCorrelação entre séries de TFI:\n", tfi_df.corr().round(3))

# Script único para diagnosticar TFI e Mapper
# - Lê seu CSV
# - Padroniza entradas (log-retornos z-score por janela)
# - Roda um grid enxuto de (epsilon, n_cubes) para o TFI
# - Salva CSVs: tfi_series.csv, tfi_summary.csv, tfi_corr.csv
# - Calcula estatísticas do Mapper (nodes/edges/components/avg_degree) por data
#   usando uma série agregada (retorno médio padronizado) — suficiente para detectar degeneração
# - Salva mapper_stats.csv
# - Gera plots simples com matplotlib
#
# Você pode mudar os parâmetros no bloco "CONFIG".
#
# Obs: Mantive o recorte em 2022 e Top15 por liquidez para rodar rápido aqui.
#      Caso queira todo o período, aumente a janela de datas e/ou TopN.

import sys, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# --------------------
# CONFIG
# --------------------
CSV_PATH = "data/ibov_b3_2019_2022.csv"
DATE_START = "2022-01-01"
DATE_END   = "2022-12-31"
TOPN_LIQUID = 15              # Top-N por ADV
WINDOW = 63                   # janela do TFI
DELAY = 1
DIM = 3
OVERLAP = 0.4
MIN_SAMPLES = 3

# Grid de sensibilidade do TFI (mantenha enxuto pra rodar rápido)
EPS_LIST = [0.05, 0.10, 0.20]
CUBES_LIST = [6, 10]

# Parâmetros do Mapper para o diagnóstico por data (um único set)
MAPPER_EPS = 0.10
MAPPER_CUBES = 10

# --------------------
# Imports do repo (tda.py)
# --------------------
sys.path.append("src/features")
tda = importlib.import_module("tda")
TFIParams = getattr(tda, "TFIParams")
tfi_score = getattr(tda, "tfi_score")
takens_embedding = getattr(tda, "takens_embedding")
mapper_graph = getattr(tda, "mapper_graph")

# --------------------
# Carregar e preparar dados
# --------------------
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
mask = (df["date"] >= DATE_START) & (df["date"] <= DATE_END)
df = df.loc[mask].copy()

# Top-N por liquidez (ADV em R$)
df["dollar_vol"] = df["close"] * df["volume"]
topn = (df.groupby("asset")["dollar_vol"]
          .mean().sort_values(ascending=False).head(TOPN_LIQUID).index.tolist())

prices = (df.pivot(index="date", columns="asset", values="close")
            .sort_index()[topn]
            .ffill()
            .dropna(how="all"))

# --------------------
# TFI para o grid de parâmetros
# --------------------
tfi_series = {}
for eps in EPS_LIST:
    for cubes in CUBES_LIST:
        params = TFIParams(delay=DELAY, dim=DIM, n_cubes=cubes,
                           overlap=OVERLAP, epsilon=eps,
                           min_samples=MIN_SAMPLES, window=WINDOW)
        ser = tfi_score(prices, params=params)
        ser.name = f"eps={eps},cubes={cubes}"
        tfi_series[ser.name] = ser

tfi_df = pd.concat(tfi_series.values(), axis=1).dropna(how="all")

# Resumo estatístico
tfi_summary = pd.DataFrame({
    "min":  tfi_df.min(),
    "max":  tfi_df.max(),
    "mean": tfi_df.mean(),
    "std":  tfi_df.std(ddof=0)
}).round(6)

tfi_corr = tfi_df.corr().round(3)

# Salvar CSVs
tfi_df.to_csv("data/test/tfi_series.csv", index=True)
tfi_summary.to_csv("data/test/tfi_summary.csv", index=True)
tfi_corr.to_csv("data/test/tfi_corr.csv", index=True)

# Mostrar resumos ao usuário
print("\n=== TFI (últimas linhas) ===")
print(tfi_df.tail(10))

print("\n=== TFI — Resumo ===")
print(tfi_summary)

print("\n=== TFI — Correlação ===")
print(tfi_corr)

# Plot das séries
plt.figure(figsize=(10,4))
for col in tfi_df.columns:
    plt.plot(tfi_df.index, tfi_df[col], linewidth=1)
plt.title("TFI — diferentes (epsilon, n_cubes)")
plt.xlabel("Data")
plt.ylabel("TFI")
plt.tight_layout()
plt.savefig("data/test/tfi_series_plot.png", dpi=150)
plt.show()

# --------------------
# Mapper: estatísticas por data (detecção de degeneração)
# Série agregada: retorno médio padronizado cross-section
# --------------------
# 1) log-retornos
logpx = np.log(prices)
rets = logpx.diff()

# 2) rolling z-score por coluna (janela=WINDOW)
rets_z = (rets - rets.rolling(WINDOW).mean()) / rets.rolling(WINDOW).std(ddof=0)

# 3) série agregada (média cross-section) — remove NaNs por linha
agg = rets_z.mean(axis=1).dropna()

# 4) para cada data t a partir de WINDOW, constrói embedding na janela [t-WINDOW+1, t]
dates = agg.index
mapper_rows = []
for i in range(WINDOW, len(dates)):
    end_date = dates[i]
    start_idx = i - WINDOW + 1
    window_vals = agg.iloc[start_idx:i+1].values  # tamanho = WINDOW

    # embedding + mapper
    try:
        emb = takens_embedding(window_vals, delay=DELAY, dim=DIM)
        G = mapper_graph(emb, n_cubes=MAPPER_CUBES, overlap=OVERLAP,
                         epsilon=MAPPER_EPS, min_samples=MIN_SAMPLES)
        import networkx as nx
        comps = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
        avg_deg = (np.mean([d for _, d in G.degree()])
                   if G.number_of_nodes() > 0 else 0.0)
        mapper_rows.append({
            "date": end_date,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "components": comps,
            "avg_degree": round(float(avg_deg), 6)
        })
    except Exception as e:
        mapper_rows.append({
            "date": end_date,
            "nodes": 0, "edges": 0, "components": 0, "avg_degree": 0.0
        })

mapper_stats = pd.DataFrame(mapper_rows).set_index("date")

# Salvar e exibir
mapper_stats.to_csv("data/test/mapper_stats.csv", index=True)
print("\n=== Mapper — Estatísticas (últimas linhas) ===")
print(mapper_stats.tail(15))

# Plots das estatísticas do Mapper
plt.figure(figsize=(10,3))
plt.plot(mapper_stats.index, mapper_stats["nodes"], linewidth=1)
plt.title("Mapper — número de nós por data")
plt.xlabel("Data"); plt.ylabel("nodes")
plt.tight_layout()
plt.savefig("data/test/mapper_nodes_plot.png", dpi=150)
plt.show()

plt.figure(figsize=(10,3))
plt.plot(mapper_stats.index, mapper_stats["components"], linewidth=1)
plt.title("Mapper — número de componentes por data")
plt.xlabel("Data"); plt.ylabel("components")
plt.tight_layout()
plt.savefig("data/test/mapper_components_plot.png", dpi=150)
plt.show()

print("Arquivos gerados:")
print("- data/test/tfi_series.csv")
print("- data/test/tfi_summary.csv")
print("- data/test/tfi_corr.csv")
print("- data/test/tfi_series_plot.png")
print("- data/test/mapper_stats.csv")
print("- data/test/mapper_nodes_plot.png")
print("- data/test/mapper_components_plot.png")
