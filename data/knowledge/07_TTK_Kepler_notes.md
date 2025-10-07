# TTK & Kepler Mapper — Notas para o Atlas (T‑HRP)

tags: [tda, ttk, paraview, persistence, reeb-graph, contour-tree, morse-smale, mapper, cover, lens, nerve, clustering, kmapper]

## 1) Tese (80/20)
- **TTK (Topology ToolKit)**: toolbox open‑source para **Análise Topológica de Dados** (com ParaView + API C++/VTK/Python), cobrindo **críticos**, **persistência/curva**, **join/split trees**, **Contour Tree**, **Morse–Smale** e **simplificação por persistência** — estruturas estáveis que resumem a “forma” de campos escalares.
- **Kepler Mapper (kmapper)**: implementação em Python do **algoritmo Mapper** — **lente → cobertura (cover) → clusterização → nervo (nerve)** — que produz um **grafo simplicial** para dados de alta dimensão. Usaremos suas **métricas de grafo** como features de **regime (TFI)** e como base de **seriação** para o HRP.

---

## 2) Workflow essencial do Mapper
1) **Projeção/Lente** (`project`/`fit_transform`): reduz dimensão ou aplica função estatística ao dado \(X\).  
2) **Cobertura (cover)** (`Cover(n_cubes, perc_overlap)`): define hipercubos em cada dimensão da lente.  
3) **Clusterização local** (por hipercubo): DBSCAN/KMeans/HDBSCAN etc.  
4) **Nervo** (`GraphNerve`): cria o 1‑esqueleto conectando clusters que compartilham pontos.  
5) **Visualização** (`visualize`): HTML/D3, com `color_values`/`node_color_function` para pintar nós.

---

## 3) API prática (KeplerMapper)
**Classe principal**: `KeplerMapper` — funções‑chave: `fit_transform`, `map`, `visualize` (constrói o grafo a partir da lente e salva um HTML).

### 3.1 `project` / `fit_transform`
- `projection`: string (`"sum"`, `"mean"`, `"knn_distance_5"`…), **obj scikit‑learn** (PCA/UMAP/TSNE) **ou índices** (`[0,1]`).  
- `scaler`: `MinMaxScaler` (padrão) ou outro; `None` desativa.  
- `distance_matrix`: pode gerar matriz de distâncias (e.g., `"cosine"`, `"euclidean"`, `"pearson"`).

### 3.2 `map(lens, X, clusterer, cover, nerve, precomputed, remove_duplicate_nodes)`
- **`lens`**: saída do `fit_transform` (\(n\times d_l\)).  
- **`X`**: dados originais para clusterização **na pré‑imagem** (recomendado); se `None`, clusteriza na lente.  
- **`clusterer`**: qualquer estimador scikit‑learn compatível (DBSCAN/KMeans/HDBSCAN).  
- **`cover`**: instância `Cover` (ver abaixo).  
- **`nerve`**: normalmente `GraphNerve()`; pode ajustar `min_intersection`.  
- **`precomputed=True`**: quando `X` for **matriz de distâncias** e o clusterer suportar `metric='precomputed'`.  
- **Retorno**: `dict` com `"nodes"`, `"links"` e `"meta"` (usado por `visualize`).

### 3.3 `visualize(graph, ...)`
- Aceita **`color_values`** (1d ou lista de 1d) e **`node_color_function`** (e.g., `'mean'`) para computar cores por nó.  
- Suporta adapters para **NetworkX** e **Plotly** (dashboards).

---

## 4) `Cover(n_cubes, perc_overlap, limits)`
- **`n_cubes`**: “resolução” por dimensão da lente.  
- **`perc_overlap`**: sobreposição adjacente; formalmente:  
  \[ \text{overlap} = \frac{|\text{cube}[i]\cap\text{cube}[i+1]|}{|\text{cube}[i]|} \]  
- **`limits`**: limites \([\min,\max]\) por dimensão (\(\infty\) cai no min/max da lente).  
- Métodos úteis: `fit` (centros), `transform` (índices por hipercubo), `transform_single`, `find`.

---

## 5) Dials recomendados (para o Atlas)
- **Lente**: `PCA(n_components=2)` (estável) para produção; `UMAP` em R&D (fixar `random_state`).  
- **Cover**: `n_cubes∈[8,20]`, `perc_overlap∈[0.1,0.5]`; gradear e **monitorar edge‑density** do grafo.  
- **Clusterer**: `DBSCAN(metric="cosine", eps≈0.3–0.7, min_samples≈3–10)`; testar `HDBSCAN` em universos grandes.  
- **Nerve**: `GraphNerve(min_intersection=1..3)` para controlar ruído em arestas.  
- **Distância precomputada**: em séries financeiras, usar **correlação/pearson** por janela com `precomputed=True` quando fizer sentido.

---

## 6) Integração no Atlas (T‑HRP)
- **TFI (0–1)**: métricas de grafo (componentes, **edge density**, grau médio, modularidade, tamanho do GCC, entropia de comunidades) → z‑scores → combinação (soft/linear) → **EWMA**.  
- **Seriação HRP**: usar embedding (PCA/UMAP da lente ou `graph_layout`) + ordenação pelo maior caminho/DFS no grafo para **quase‑diagonalizar Σ** antes da bisseção do HRP.  
- **Risk controls**: mapear TFI→`target_vol`/`gross`/caps com **histerese** (já implementado no `engine.py`).

---

## 7) Pitfalls & mitigação
- **Sobre/under‑fragmentação** do grafo: ajuste `n_cubes`×`perc_overlap`×`eps`; monitore monotonicidade de **edge density**.  
- **Lente instável (t‑SNE)**: prefira **PCA/UMAP**; fixe seeds.  
- **Duplicação de nós**: habilite `remove_duplicate_nodes`.  
- **Custo**: subamostrar \(X\) para o grafo e **jackknife** nas métricas; cache da lente.  
- **Dependência do clusterer**: valide com dois clusterers (DBSCAN vs. KMeans) e meça robustez do TFI.

---

## 8) Snippet mínimo (copie‑e‑cole)
```python
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

mapper = km.KeplerMapper(verbose=1)
lens = mapper.fit_transform(X, projection=PCA(n_components=2))  # ou mapper.project(...)

cover = km.Cover(n_cubes=12, perc_overlap=0.3)
graph = mapper.map(lens, X=X,
                   clusterer=DBSCAN(metric="cosine", eps=0.5, min_samples=5),
                   cover=cover,
                   nerve=km.GraphNerve(min_intersection=1),
                   precomputed=False,
                   remove_duplicate_nodes=True)

html = mapper.visualize(graph,
                        path_html="mapper_output.html",
                        color_values=returns_last_month,  # ex.: pintar por retorno
                        node_color_function="mean")
```

---

## 9) Checklist de validação
- [ ] **Edge‑density** varia de forma **monotônica** ao mudar `n_cubes`/`perc_overlap`/`eps`.  
- [ ] **TFI** aumenta em janelas de **alta vol/MDD** e cai em regimes calmos.  
- [ ] Seriação por Mapper reduz **off‑diagonal** de Σ e **turnover** vs. ordem aleatória.  
- [ ] Robustez a **clusterers** (DBSCAN/HDBSCAN/KMeans): ranking do TFI preservado.  
- [ ] Impacto na carteira: melhor **Calmar** e **MDD** sem piorar **capacidade**.

---

## 10) Referências rápidas
- **Docs Kepler Mapper**: API (`KeplerMapper`, `Cover`, `GraphNerve`), Getting Started, Visuals/Adapters.  
- **Paper JOSS (2019)**: “Kepler Mapper: A flexible Python implementation of the Mapper algorithm”.
