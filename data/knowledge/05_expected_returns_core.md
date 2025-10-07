# Expected Returns (Antti Ilmanen) — Núcleo Operacional para o Atlas (T‑HRP)

> 80/20 do livro aplicado ao seu projeto **Atlas** (T‑HRP com TDA e gestão de risco ativa). Texto em PT‑BR, direto ao ponto e orientado a implementação.

## 1) Ideias‑chave (o que é “não negociável”)

- **Esperados > históricos:** priorize medidas **ex‑ante** (yields, valuations, carry) em vez de extrapolar médias históricas.
- **Diversificação de *fontes* de retorno:** combine **classes**, **estilos** (value, carry, momentum, vol) e **fatores** macro (crescimento, inflação, iliquidez, caudas). Use isso para **tiltar** o Atlas sem perder o DNA de risco-paridade.
- **Prêmios variam no tempo:** ERPs/BRPs/CRPs são **cíclicos**; deixe o modelo reagir a **regimes** (inflação x crescimento, safe‑haven, crises).
- **Sinais simples funcionam:** *carry* (payout yield), *value* (E/P, CAPE), *slope* da curva, *credit spreads*, “CP‑factor” (Cochrane‑Piazzesi), *survey‑based* BRP, etc.
- **Evite *chasing* e overfit:** combine sinais, **normalizados & regularizados**, com *walk‑forward* + **Purged K‑Fold** e **embargo temporal**.
- **Função do ativo no portfólio:** ações = crescimento; Treasuries = **hedge** de deflação/safe‑haven; crédito = spread/iliquidez; *alts* = carry/valor/trend.
- **Integração prática:** use sinais **só com informação disponível na data**, com *lags* realistas e restrições de *turnover/capacidade*.

## 2) Sinais ex‑ante priorizados (o que calcular no `factors.py`)

### 2.1 Ações (ERP)
- **Carry/Payout**: `NetPayoutYield = DividendYield + BuybackYield – NetIssuance`.
- **Valuation**: `EarningsYield (trailing/forward)`, `CAPE‑E/P`.
- **DDM / Equity‑Bond Premium (ERPB)**: `ERPB = (D/P + G) – Y`, com *G* (crescimento real de longo prazo) vindo de média de pesquisas + janelas longas de GDP/earnings.
- **Cross‑checks**: *Fed model* (`E/P – Y`), indicadores de risco (vol implícita), e *breadth* (amplitude de avanços).

### 2.2 Títulos Soberanos (BRP)
- **Ex‑ante real yield** (linkers/inflação esperada) como *driver* de horizontes longos.
- **Slope 10y–3m** para tático (T+1q / 1y).
- **Cochrane‑Piazzesi BRP** (combinação de forwards/Ytm) e **BRP por pesquisas** (encostas + expectativas).
- **Safe‑haven state**: *stock‑bond correlation* negativa; *flight‑to‑quality*; vol de ações alta.
- **Volatilidade de bonds** e incerteza de inflação como *moduladores* de risco.

### 2.3 Crédito (CRP)
- **Spreads** (IG/HY), *OAS*, *short‑duration IG pocket*, *fallen angels*.
- **Liquidez/Oferta**: *debt maturity share*, *new issuance*; proxy de fluxo (ETF primário/secundário).

### 2.4 Commodities/FX/Alternativos
- **Carry** (curva *backwardation/contango*, *basis*), **valor** (PPP / *real exchange rate mispricing*), **momentum/trend** (12‑1) robuste.
- **Vol‑carry**/**insurance selling** (atenção a *left‑tail*).

## 3) Regimes e condicionantes (sinais de estado)

- **Inflação x Crescimento**: deflação/baixo‑estável/alta; associe a **correlação ações‑bonds** e ao *hedge value* dos Treasuries.
- **Crises (safe‑haven)**: *drawdowns* de ações + salto em vol → *tilt* para duration/qualidade.
- **Ciclo**: ISM, desemprego, *CFNAI*, lucros/PIB.
- **Oferta & Demanda**: *debt maturity share* dos Treasuries, demografia (mais lento, só para *priors*).

## 4) Integração no Atlas (arquivos e *hooks*)

- `factors.py` — **NOVOS** *features*:
  - `net_payout_yield(df_prices, df_divs, df_buybacks, df_issuance)`
  - `earnings_yield(series_eps, method="trailing|forward|CAPE")`
  - `ddm_erpb(D_over_P, G_real, Y_bond)`  → retorna `{ex_ante_equity_real, ERPB}`
  - `term_slope(y10, y3m)`, `cp_brp(yields_or_forwards)`, `survey_brp(surveys)`
  - `bond_real_yield(y_nom, exp_infl)`, `bond_vol(realized)`
  - `stock_bond_corr(window)` (para *state* de safe‑haven)
  - `credit_spread(oas_ig, oas_hy)`, `debt_maturity_share(treasury_issuance)`

- `risk_controls.py` — **tilt previsional com limites**:
  - `expected_sharpe_tilt(w_hrp, mu_signals, Sigma, max_tilt=0.20, by_cluster=True)`
    - mistura **HRP puro** com um vetor de retornos esperados (**mu_signals**) com *box constraints* por **cluster topológico**.
  - **Regime gating**: só aplica tilt quando *sinal z‑score* ≥ *t* e *drawdown* < limite.

- `hrp.py` / `tda.py` — sem mudar lógica básica; exponha `prior_mu` (opcional) p/ “HRP‑tilt”:
  - `w = blend(w_hrp, argmax_{w} w' mu - λ w' Σ w,  ||w - w_hrp||_∞ ≤ ε)`

- `walk_forward.py` / `purged_cv.py` — valide os *features* com **Purged K‑Fold + Embargo** e *walk‑forward* rolante.
- `kpis.py` — adicione **ERP/BRP/CRP *nowcasts*** aos *dashboards* e *breakdown* por cluster.
- `robustness.py` — *slicing* por **regimes de inflação** e **sinal de correlação ações‑bonds**.

## 5) Especificações dos sinais (fórmulas e cuidados)

### ERP via DDM (ex ante)
- **Equity–Bond Premium:** `ERPB = (D/P + G) – Y` (consistente em termos **reais**).  
- **G**: média combinada (pesquisas *long‑run real GDP/earnings* + janelas decenais de crescimento observado).  
- **Limpeza**: incluir **repurchases** e **emissões** → *Net Payout*. Evitar incluir M&A (reduz previsibilidade).  
- ***Lag***: use o *release* mais recente *disponível na data*. Documente *lags*.

### BRP (bonds)
- **Curto prazo (T+1q/1y)**: `Slope(10y-3m)`, `CP‑BRP`, `Kim‑Wright/Survey‑BRP`.
- **Longo prazo (3‑5y)**: `RealYield(ex‑ante)` domina; *slope* como *proxy* estrutural é fraco.  
- **Regimes**: *stock‑bond corr* < 0 → **hedge premium** e valuations esticadas; > 0 → risco de perder *hedge*.  

### Crédito
- **Spreads** preditivos; **maturity**/qualidade importam; inclua *liquidity* e *supply* (`debt maturity share`).

## 6) Pipeline (padrão Atlas)

1. **Ingest**: séries (preços, dividendos, buybacks, emissões, EPS, *yields*, curvas, surveys, OAS).  
2. **Feature Engine** (`factors.py`): z‑score/robust‑winsor, *lags* consistentes.  
3. **Modelos de *nowcast*** (`engine.py`): combinação simples (média ponderada/bayesiana) → vetor `mu_signals`.  
4. **Alocação** (`hrp.py`): HRP base → **tilt** previsional com limites por cluster TDA.  
5. **Execução & Risco** (`execution.py`, `risk_controls.py`): *turnover*, *leash*, *drawdown guard*, *vol‑target*.  
6. **Validação**: *walk‑forward* + **Purged CV**; *slicing* por regime.  
7. **KPIs**: ex‑ante ERPs/BRPs/CRPs, *hit‑rate* de sinais, *turnover/custos*, estabilidade por cluster.

## 7) Pseudocódigo (exemplos mínimos)

```python
# factors.py
def ddm_erpb(D_over_P, G_real, Y_bond):
    ex_ante_equity_real = D_over_P + G_real
    erpb = ex_ante_equity_real - Y_bond
    return dict(ex_ante_equity_real=ex_ante_equity_real, erpb=erpb)

def expected_sharpe_tilt(w_hrp, mu, Sigma, max_tilt=0.20, by_cluster=None):
    # solve: max w' mu - λ w' Σ w, s.a. ||w - w_hrp||_inf ≤ max_tilt
    # e opcionalmente agrupar por cluster e limitar tilt agregado
    ...
```

## 8) Checklist de implementação (para não esquecer)

- [ ] **Net Payout** (com *buybacks – issuance*).  
- [ ] **DDM (real)** com *G* de pesquisas + janelas longas.  
- [ ] **CP‑BRP, Survey‑BRP, RealYield**.  
- [ ] **State**: correlação ações‑bonds; regimes de inflação.  
- [ ] **Tilt** limitado por cluster (TDA) e *gating* por *z‑scores* e *drawdown*.  
- [ ] **WF + Purged CV + Embargo** nos *features* e no tilt.  
- [ ] **KPIs**: agora‑casts de premia + *attribution* por cluster.

---

**Observação:** Este núcleo só usa informações ex‑ante e traz limites explícitos para manter o caráter conservador do T‑HRP. Complementa o `ml4am_core.md` sem redundância.

