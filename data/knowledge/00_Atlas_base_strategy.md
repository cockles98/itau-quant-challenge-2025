Perfeito, Felipe — abaixo está a **T-HRP v3.0 (Topological-HRP Plus)**, já **refeita** para incorporar tudo que faltava: overlay de fatores, *vol targeting*, *ATR sizing*, *purged/embargoed CV*, regras de risco/execução, testes de robustez e capacidade. Mantive o **núcleo topológico \+ HRP** e o **walk-forward**/custos que você já havia estruturado. Onde eu referencio algo que já está no seu PDF, cito; o restante é complemento novo.

---

# **Atlas - Uma Estratégia Topológica com Gestão de Risco Ativa, Lendo o Mercado Como um Mapa**

## **0\) Objetivo**

Construir uma carteira sistemática, **long-only**, com **alocação HRP** adaptada por **sinais topológicos de regime** e **overlay de fatores clássicos**, sob **gestão ativa de risco** (alvo de volatilidade e dimensionamento por ATR), com **validação walk-forward** rigorosa, **custos/slippage realistas** e **limites operacionais**.

---

## **1\) Universo & Dados**

* **Universo base**: Top-20 por liquidez do universo do desafio (mesma premissa do seu doc), com filtros mínimos de **ADV**, preço e *age* do ativo.

* **Histerese**: entrada/saída exigem **2 rebalanceamentos consecutivos** para evitar *churn*.

* **Frequência**: diária (ou intradiária se permitido) com **rebalance mensal** (mantido do seu plano).

---

## **2\) Features / Sinais**

### **2.1 Sinais Topológicos de Regime (núcleo)**

* **Reconstrução de fase (Takens)** → *embedding* com *delay* e dimensão ótimos.

* **KeplerMapper/TDA** → grafo de vizinhança, **clusterização** e **Topological Fragility Index (TFI)** como *regime score*.

* **Uso**: *score* de regime alimenta pesos de *risk budget* por cluster/ativo (ver §4). (Arquitetura/justificativa constam no PDF.)

### **2.2 Overlay de Fatores “clássicos”**

* **Momentum** (12–1, e/ou *slope* n-dias).

* **Value/Quality** (proxies simples se disponíveis; caso não, use estabilidade de volatilidade, drawdown recente e *trend persistence* como *quality-like*).

* **Combinação**: *score* final \= `α·(TFI/Regime) + β·Momentum + γ·Quality`, com `α>β≥γ` (prioriza regime). **Normalizar** e **winsorizar** sinais antes de combinar.

---

## **3\) Construção da Carteira**

### **3.1 HRP Topológico (já definido)**

* **Seriação/quase-diagonalização** da covariância segundo a hierarquia do grafo TDA;

* **Bisseção recursiva** com ponderação por variância inversa (Riskfolio-Lib).

* **Racional**: HRP evita inversão de Σ (estável OOS) e aqui herda **clusters** do grafo, adaptando-se a mudanças de estrutura.

### **3.2 Pesos “target” por sinal**

* Em cada **sub-cluster HRP**, distribua *risk budget* proporcional ao **score combinado** (regime \+ fatores), com **cap de concentração** por ativo/cluster (ver §5).

---

## **4\) Gestão de Risco Ativa**

### **4.1 Volatility Targeting (carteira)**

* Alvo de vol anualizada: **10%** (ajustável).

* A cada rebalance **escale os pesos** para que a vol realizada (janela 60d) ≈ alvo. Se vol-realizada \> **1,5×** alvo intra-mês, aplique *clip* proporcional.

### **4.2 ATR Sizing (por ativo)**

* Dentro de cada sub-cluster, **normalizar risco**: `peso_i ∝ 1 / ATR_i(14)` (ou 20\) → ativos mais voláteis recebem peso menor.

* **Resultado**: uniformiza contribuição de risco e reduz MDD em cenários turbulentos.

---

## **5\) Regras de Risco, Limites e Execução**

* **Max weight por ativo**: **10%**; **por cluster**: **35%**.

* **Turnover cap** por rebalance: **≤25%** do portfólio.

* **Exposição**: 0–100% long-only; caixa residual permitido.

* **Stops/Kill-switch**: se **MDD rolling-90d \< −X%** ou **vol-realizada \> 1,8×** o alvo, **de-risk → cash**; reentrada quando TFI/vol voltarem abaixo de limiares.

* **Capacidade/Liquidez**: cada ordem ≤ **5% do ADV**; *lot size* mínimo, *no-trade* em *spreads* extremos.

**Custos/Execução** (preservados e detalhados):

* **Comissão**: **10 bps**; **slippage** cresce com participação no volume (função do tamanho/ADV). **Backtest** com **rebalance mensal**; ambiente `backtrader`.

---

## **6\) Backtesting & Validação**

### **6.1 Walk-forward (mantido)**

* Janelas rolantes: **2 anos IS / 6 meses OOS**, avançando 6 meses; retreina T-HRP e aplica OOS travado.

### **6.2 Purged / Embargoed CV (novo)**

* Para *tuning* de hiperparâmetros **TDA/KeplerMapper** (n\_cubes, overlap, delay, dim) e do **mix α,β,γ**, use **Purged K-Fold** com **embargo temporal** para eliminar *look-ahead*.

* Avalie **robustez** em *grid* amplo; reporte **heatmaps** de Sharpe/IR vs. hiperparâmetros.

---

## **7\) Métricas & Benchmarks**

* **KPIs**: Sharpe, Sortino, **CAGR**, Vol, **MDD**, Calmar, **Turnover** (você já listou).

* **Benchmarks**: **HRP tradicional**, **1/N**, **índice cap-weighted** (mantidos).

* **Adicionais**: *Hit-rate*, *average trade P\&L*, *ex-ante* risk contribution, *capacity curve* (P\&L vs. participação no ADV).

---

## **8\) Robustez & Estresse (novo)**

* **Robustez de parâmetros**: *heatmaps* de desempenho vs. (n\_cubes, overlap, delay, dim, janela de vol, ATR-len).

* **Estresse temporal**: subperíodos por regime (alto/baixo TFI), *pre/post* eventos macro.

* **Perturbação de custos**: multiplique comissões/slippage por {0,5×, 1×, 2×}.

* **Ruído de dados**: *jitter* pequeno nos *returns* para avaliar fragilidade.

---

## **9\) Capacidade (novo)**

* **Regra**: limitar *participation rate* a **≤5% ADV**; simular *market impact* simples (*square-root* ou *piecewise*).

* **Curva de capacidade**: para cada *participation cap*, estime Sharpe/IR e MDD; selecione *sweet spot* que preserva edge.

---

## **10\) Pseudocódigo (execução mensal)**

for rebalance\_date in schedule(monthly):  
    universe \= top\_liquidity(N=20, filters=ADV\_min, price\_min, age\_min, with\_hysteresis=2)  
    X \= build\_time\_series(universe)  
    \# 1\) Sinais  
    reg\_score \= TFI\_TDA\_KeplerMapper(X, params\_TDA)             \# regime/topologia  
    mom \= momentum\_signal(X, lookbacks=\[63,126,252\])  
    quality \= quality\_proxy(X)                                   \# se disponível  
    score \= normalize(α\*reg\_score \+ β\*mom \+ γ\*quality)  
    \# 2\) HRP Topológico  
    Σ \= cov\_matrix(X, lookback=cov\_lb)  
    order \= topo\_seriation(Σ, graph\_from\_TDA)  
    w\_hrp \= hrp\_weights(Σ, order)                                \# Riskfolio-Lib  
    \# 3\) Risco Ativo  
    w\_signal \= apply\_signal\_budget(w\_hrp, score, caps={asset:10%, cluster:35%})  
    w\_atr    \= atr\_risk\_normalize(w\_signal, ATR\_len=14)  
    w\_target \= scale\_to\_vol(w\_atr, target\_vol=10%, window=60d)  
    \# 4\) Limites & Execução  
    w\_final  \= apply\_turnover\_cap(w\_target, 25%)  
    trades   \= roll\_from\_to(prev\_weights, w\_final, costs=10bps, slippage=f(size/ADV))  
    if kill\_switch\_triggered(MDD\_90d, real\_vol): de\_risk\_to\_cash()

---

## **11\) Roadmap de Entrega**

1. **MVP (1–2 dias úteis)**: overlay de **momentum** \+ **vol target** \+ **ATR sizing** sobre seu pipeline atual; validação *walk-forward* igual à sua.

2. **Validação (3–4 dias)**: **purged/embargoed CV** para hiperparâmetros TDA/Kepler; *heatmaps*; *stress* de custos.

3. **Operacional (1–2 dias)**: limites de risco/turnover, **capacity check** (5% ADV) e **kill-switch**.

---

## **12\) O que permanece do seu documento original**

* **Tese topológica \+ HRP** (espinha dorsal) e narrativa de inovação.

* **Ambiente e *walk-forward***; **custos/slippage**; **rebalance mensal**; **KPIs/benchmarks**.

---

### **Entregáveis (se quiser, já te devolvo o PDF pronto)**

* Seções **2, 4, 5, 6.2, 8, 9** atualizadas no seu relatório (“v3.0”).

* **Apêndice** com *heatmaps* de robustez e *capacity curve*.

* *Notebook* com funções `scale_to_vol`, `atr_risk_normalize`, `purged_kfold_time_series`.  
