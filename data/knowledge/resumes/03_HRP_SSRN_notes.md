# HRP (SSRN) — Notas para o Atlas (T‑HRP)

tags: [hrp, clustering, seriation, ivp, covariance, denoise, turnover]

## 1) Tese do paper (80/20)
O **Hierarchical Risk Parity (HRP)** resolve a alocação em portfólios sem inverter a matriz de covariância, evitando instabilidade de média‑variância (Markowitz) e tornando os pesos **mais estáveis OOS**. A ideia central é:
1) medir **distâncias** entre ativos a partir da correlação (ou métrica robusta);
2) **clusterizar** (árvore) e **ordenar** os ativos por proximidade (seriação);
3) **quase‑diagonalizar** Σ (ativos próximos ficam adjacentes);
4) **alocar por bisseção**, usando **variância inversa (IVP)** em cada subcluster.

Resultado: risco mais homogeneamente distribuído e menor turnover, especialmente em amostras pequenas/ruidosas.

---

## 2) Algoritmo HRP (passo‑a‑passo)
**Entrada**: retornos \(R_{t,i}\) e janela de covariância \(W\).  
**Saída**: pesos \(w_i \ge 0\), \(\sum w_i = 1\).

1) **Covariância/Correlação**
- \(\Sigma = \operatorname{Cov}(R)\) (rolling, janela \(W\)).
- \(ho_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)\).

2) **Distância** (padrão do paper)
- \(D_{ij} = \sqrt{	frac{1}{2}(1-ho_{ij})}\).
  - Alternativas robustas (no Atlas): VI/MI ou detonar \(PC_1\) antes.

3) **Clustering & Seriação**
- Construa uma árvore (ex.: single‑linkage) e derive uma **ordem linear** (seriação) que coloque ativos similares juntos.
- No Atlas, a ordem pode vir do **grafo topológico** (TDA) → `topo_seriation_from_graph`.

4) **Quase‑diagonalização**
- Reordene \(\Sigma\) pela seriação para concentrar covariâncias fora da diagonal em blocos.

5) **Alocação Recursiva (IVP)**
- Faça **bisseção** da ordem \(\mathcal{O}\) em dois subgrupos \(L\) e \(R\).
- Variância de cluster (aproximação IVP):
  \[
  \sigma^2(C)= rac{1}{|C|^2}\mathbf{1}^	op \Sigma_{C,C}\mathbf{1}
  \]
- Alocação entre blocos:
  \[
  a_L = 1 - rac{\sigma^2(L)}{\sigma^2(L)+\sigma^2(R)},\quad
  a_R = 1-a_L
  \]
- Replique o processo recursivamente até nós folha; normalize \(w\).

---

## 3) Dials (alavancas que mais importam)
- **Janela de covariância \(W\)**: 60–126 dias costumam equilibrar ruído e responsividade.
- **Denoise/Detone**: Marčenko–Pastur + remoção do **1º PC** (fator de mercado) melhoram estabilidade.
- **Distância**: 1‑corr é padrão; **VI/MI** podem ser mais robustas em regime não linear.
- **Leaf ordering**: single‑linkage com *optimal leaf ordering* ou seriação vinda do **grafo TDA**.
- **Rebalance freq.**: mensal (ou menor) para conter turnover; combine com **cap de turnover**.
- **Sizing**: pós‑HRP aplicar **ATR** e **target vol** (no Atlas) para nivelar risco.

---

## 4) Pitfalls (e como evitar)
- **Σ mal condicionado**: use **denoise** e descarte colunas com histórico insuficiente.
- **Clustering instável**: bootstrap/heatmap de estabilidade; TDA pode fornecer **ordens mais persistentes**.
- **Concentração em um cluster**: imponha **caps** por ativo/cluster e monitore **contribuição de risco**.
- **Fuga para 1/N** se sinais forem fracos: valide se HRP realmente melhora vs. 1/N e cap‑weighted.
- **Look‑ahead**: alimente o HRP apenas com dados **rolantes** até \(t\); valide com **walk‑forward** + **Purged/Embargoed CV**.

---

## 5) Mapa para o código (Atlas)
- `hrp.py`
  - `rolling_cov(returns, window=W)`: covariâncias rolantes estáveis.
  - `topo_seriation_from_graph(cov, graph)`: ordem vinda do grafo (TDA) ou fallback para ordem natural.
  - `hrp_weights_from_order(cov, order)`: bisseção recursiva + IVP e normalização.
- `position_sizing.py`
  - `atr_risk_normalize` → distribui risco intra‑cluster.
  - `scale_to_vol` → bate **target vol** da carteira.
- `engine.py`
  - Orquestra **rebalance mensal**, **turnover cap**, **caps** por ativo/cluster e **kill‑switch**.
- `robustness.py`
  - Heatmaps de sensibilidade de \(W\) e da métrica de distância/seriação.

---

## 6) Checklist de validação (rápido)
- [ ] HRP > 1/N em **Calmar** e **MDD** em várias janelas.
- [ ] **Turnover** controlado e estável OOS.
- [ ] Pesos **não explodem** quando um ativo muda de regime.
- [ ] **Stress** de custos (0.5×/1×/2×) mantém ranking das versões.
- [ ] **Regimes TDA** mudam a seriação/pesos de forma coerente (diagnóstico de clusters).

---

## 7) O que levar para produção
- Artefatos: pesos por data, cov rolante, ordem/árvore e métricas de **risk contribution**.
- Monitores: **HHI** de pesos, **exposição por cluster**, drift de Σ, **PSR/deflated Sharpe**.
- Guard‑rails: **kill‑switch** por MDD/vol, **cooldown** e histerese de reentrada.

---

## 8) Referências (essencial)
- López de Prado, “Building Diversified Portfolios that Outperform Out of Sample” (SSRN, 2016/2020).  
- Teses/artigos de replicação HRP; notas do Atlas em `code_crosswalk.md` e `00_curriculum.md`.
