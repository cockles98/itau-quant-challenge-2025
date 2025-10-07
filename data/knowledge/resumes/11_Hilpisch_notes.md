# Python for Finance — Notas de estudo
**Referência:** Yves Hilpisch — *Python for Finance: Mastering Data‑Driven Finance* (O’Reilly, 2ª ed., 2018).

> **TL;DR:** O livro é um **manual prático** para finanças quantitativas em Python: começa por **NumPy/pandas/visualização**, progride para **estatística e machine learning**, depois **backtesting/execução**, e fecha com **modelos estocásticos, simulação de Monte Carlo e precificação de derivativos**. O foco é “**código antes de teoria pesada**”, com ênfase em **vetorização**, **validação adequada** e **integração ponta‑a‑ponta** (dados → sinal → portfólio → execução → avaliação).

---

## 1) Ideias‑chave (para quant)
- **Dados & vetorização primeiro:** operações NumPy/pandas bem projetadas são ordens de grandeza mais rápidas que loops puros; **evite `for`** em caminhos críticos.
- **Pure‑Python vs. aceleração:** perfis → tente **broadcasting**, **numexpr**, **Numba**, **Cython** quando necessário; só otimize gargalos reais.
- **Time series financeiras ≠ séries estáticas:** **marcações de tempo irregulares**, feriados e *rolls* de contratos; **resampling** e **joins** cuidadosos.
- **Backtesting honesto:** separação temporal clara, **no look‑ahead**, **no leakage**; **event‑driven backtests** para ordens/latência e **vectorized backtests** para protótipos rápidos.
- **Risco e custos são “parte do modelo”:** *slippage*, *fees*, impacto de mercado e restrições (alavancagem, *cash drag*, *shorting*) devem estar na simulação e nos KPIs.
- **Modelos estocásticos como geradores de cenários:** GBM, saltos (Merton), volatilidade estocástica; **simulações Monte Carlo** para precificação e *risk* (P&L, VaR/ES).

---

## 2) Mapa do conteúdo (por partes)
**Parte I — Fundamentos de Python científico**
- **NumPy** (arrays, *broadcasting*, *ufuncs*, geração de aleatórios), **pandas** (Series/DataFrame, *index* por datas, *groupby*, *rolling*, *resample*), **matplotlib**/**plotly** para gráficos.
- **I/O e data wrangling:** CSV/HDF5/SQL, *tidy data*, *merge/join/concat*, *pipelines* reproduzíveis.
- **Desempenho:** *vectorization*, *memory layout*, *views vs. copies*, *profiling*, **Numba** para *loops quentes*.

**Parte II — Estatística e ML aplicados**
- **Estatística descritiva & testes** (normalidade, *stationarity*, autocorrelação, heteroscedasticidade).
- **Modelos lineares e regularização** (OLS, ridge, lasso), **árvores/ensembles** e **SVM** para *tabular*; *cross‑validation* e *walk‑forward*.
- **Métricas específicas** para *trading*: *hit ratio*, *payoff ratio*, *turnover*, **information ratio** (IR), **Sharpe**, **max drawdown**.

**Parte III — Backtesting & execução**
- **Arquitetura event‑driven:** *data handler* → *strategy* → *portfolio* → *execution handler* → *broker* → *performance*.
- **Execução:** ordens *market/limit/stop*, *position sizing*, **custos e impacto**; *paper trading* antes da produção.
- **Automação:** *schedulers*, *logging*, *checkpointing*, *config management*.

**Partes IV & V — Modelagem estocástica e derivativos**
- **Processos**: **GBM**, **OU**, **CIR**, **jump‑diffusion**; discretizações de **Itô** (Euler‑Maruyama, Milstein).
- **Monte Carlo**: amostragem, *antithetic*, *control variates*, *Brownian bridges*; **LSMC** (*Least Squares Monte Carlo*) para opções americanas.
- **Precificação**: Black‑Scholes e *Greeks*, binomial/trinomial, PDEs e equivalência com MC em risco‑neutro.
- **Portfólios de derivativos**: agregação, *greeks* de carteira, *stress*, *what‑if*.

---

## 3) Receitas úteis (pseudocódigo)
**Vectorized backtest (long‑only)**
```
weights_t = f(signals_t; constraints)
returns_t = weights_{t-1} · asset_ret_t - costs_t(weights_{t-1}→weights_t)
equity_t  = equity_{t-1} * (1 + returns_t)
KPIs = sharpe(equity), max_dd(equity), calmar(...), turnover(...)
```

**Event‑driven (esqueleto)**
```
while trading_day:
    events = data_handler.update()
    for ev in events:
        strategy.on_event(ev)          # gera sinal/ordem
        portfolio.on_order(order)      # risco, sizing, cash
        broker.execute(order)          # preços/custos/latência
    performance.update()
```

**Monte Carlo (GBM com controle de variância)**
```
for path in 1..N:
    W = brownian_bridge(T, steps)
    S = S0 * exp((r - 0.5*sigma^2)*t + sigma*W)
    payoff[path] = disc * g(S)         # preço ou P&L
price = mean(payoff * control_variate_adjustment)
se    = std(payoff)/sqrt(N)
```

---

## 4) Como “plugar” no nosso projeto (crosswalk)
- **`loaders.py`** → leitura de CSV/Parquet/HDF5; *resample*; *adjustments* (splits, *roll* de futuros).
- **`factors.py`** → construção vetorizada de *features* (retornos, *momentum*, *vol*, *carry*, *roll‑down*, TDA); janelas com `rolling().apply()` eficientes.
- **`walk_forward.py`**/**`purged_cv.py`** → *walk‑forward* e **Purged K‑Fold**; **a separação temporal do Hilpisch** casa com AFML.
- **`risk_controls.py`**/**`position_sizing.py`** → alvos de risco (vol target), *Kelly parcial*, *leverage caps*, *position limits*.
- **`hrp.py`**/**`robustness.py`** → alocação **HRP/MinVar**; *bootstrapping* de correlações e *clustering* estável.
- **`execution.py`**/**`costs.py`**/**`capacity.py`** → custos lineares/não‑lineares, *slippage*, impacto (*square‑root law*), limites de capacidade.
- **`kpis.py`** → **Sharpe/Sortino/IR**, *turnover*, *drawdown*, *hit/payoff*, **PSR** (*Probabilistic Sharpe Ratio*).
- **`engine.py`**/**`main.py`** → arquitetura **event‑driven** inspirada no livro: *handlers*, *queues*, *broker simulado*, *logging*.

---

## 5) Armadilhas frequentes
- **Look‑ahead/leakage**: *joins* em `pandas` que cruzam *labels* (ex.: usar retorno futuro na regressão) → sempre alinhar com *lag* explícito.
- **Rebalance sincrônico irrealista**: ignorar horários, *latência* e *partial fills* distorce métricas.
- **Custos subestimados**: *spread* dinâmico, *market impact* e *fees* fixas… **modele por classe de ativo e regime**.
- **Overfitting por “grid hunting”**: *sweeps* de hiperparâmetros extensos sem **nested CV** → use *early stopping* e **regra de 1‑SE**.
- **Uso ingênuo de Monte Carlo**: amostragem sem *variance reduction* exige N impraticável; combine técnicas e pare quando **erro padrão** atingir alvo.

---

## 6) Checklist de implementação
- [ ] Pipelines reprodutíveis (semente, versão de dados, *config*).
- [ ] Validação temporal: **Purged/Embargo** e **walk‑forward**.
- [ ] KPIs padrão e *report* automático (PDF/HTML).
- [ ] Módulo de custos e restrições ativo por default.
- [ ] *Backtest vectorized* para POC + *event‑driven* para realismo.
- [ ] *Profiling* (line‑profiler) e aceleração seletiva (Numba).
- [ ] Suítes de **robustez** (bootstraps, *stress*, *sensitivity*).

---

## 7) Fórmulas/lembranças rápidas
- **GBM:** \( dS_t = \mu S_t dt + \sigma S_t dW_t \) → \( S_T = S_0\,\exp\{(\mu-\tfrac{1}{2}\sigma^2)T + \sigma W_T\} \).
- **OU:** \( dX_t = \kappa(\theta - X_t)dt + \sigma dW_t \). **CIR:** \( dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}\,dW_t \).
- **Black‑Scholes (call):** \( C = S_0\Phi(d_1) - Ke^{-rT}\Phi(d_2) \), \( d_1 = \frac{\ln(S_0/K)+(r+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} \).
- **LSMC (esboço):** regressione *payoffs* descontados futuros em *bases* de \(S_t\); exerça quando valor imediato > valor de continuação estimado.

---

## 8) Relações com outros textos do /knowledge
- **AFML**: purging/embargo, *meta‑labeling* e desenho de CV (conexão direta com nosso `purged_cv.py`).
- **ESL**: *bias–variance*, regularização e *ensembles* → guia para os modelos do Cap. de ML do Hilpisch.
- **APM (Grinold & Kahn)**: risco, *IR*, *breadth* e *transfer coefficient* para avaliar *alpha* treinado em Python.
- **Hull/Almgren‑Chriss**: risco de opções e execução ótima; integram com módulos de **custos** e **derivativos**.

---

## 9) Mini‑roteiro de estudo (10 horas)
1. **2h** NumPy/pandas avançado + I/O.
2. **2h** Estatística básica + validação temporal.
3. **2h** Backtesting vetorizado + KPIs + custos.
4. **2h** Event‑driven + execução.
5. **2h** Monte Carlo + Black‑Scholes + LSMC.

---

> **Resumo executivo:** Foque em **pipelines vetorizadas, validação temporal correta e modelagem explícita de custos/risco**. Use Monte Carlo e modelos estocásticos para **gerar cenários** e **precificar**, mantendo a arquitetura **event‑driven** para realismo de execução.
