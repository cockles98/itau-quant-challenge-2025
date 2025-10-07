
# AFML → Atlas (T-HRP) — Context Pack

## 1) Ideia central
O livro ensina um pipeline de ML financeiro robusto contra overfitting e vazamento temporal, desde a **construção de amostras** até **validação**, **dimensionamento de apostas (bet sizing)** e **alocação de portfólio**. O foco é transformar séries financeiras ruidosas em dados mais informativos e testáveis.

## 2) Pipeline resumido (ordem sugerida)
1) **Barras** informacionais (dólar/tick/volume/imbalance/run) → reduzem *Epps effect* e heterocedasticidade.  
2) **Filtros de evento** (CUSUM por volatilidade) → definem *candidate events*.  
3) **Rotulagem** (Triple-Barrier + meta-labeling) → sinais 1/0/-1 com *vertical barrier*.  
4) **Pesos amostrais** (overlap & returns/vol) → corrigem concorrência de rótulos.  
5) **Fractional differencing** (d≈0.2–0.6) → torna séries quase-estacionárias preservando memória.  
6) **Features** (microestrutura, estatísticas robustas, TDA, fatores) + *de-meaning by regime*.  
7) **Validação**: *Purged K-Fold* + *embargo* e **Combinatorial Purged CV** (CPCV).  
8) **Tuning** (grid/bayes) com scorings financeiros (e.g., *utility*, drawdown-aware).  
9) **Importância de features** (MDA / SFI) e estabilidade.  
10) **Bet sizing** (Kelly fracionário, *signals → size*) com **meta-labeling**.  
11) **Alocação**: HRP/THRP com *vol targeting*, *leverage caps* e *turnover control*.  
12) **Backtest robusto**: *walk-forward*, custos, *slippage*, capacidade e *risk controls*.  
13) **Produção**: monitoramento (PSR, HHI exposures, drift), *kill-switches* e *post-trade attribution*.

## 3) Algoritmos — lembretes operacionais

### Barras informacionais
- **Dollar bars**: fecha barra quando soma(|preço*volume|) ≥ alvo em moeda.  
- **Imbalance/Run bars**: fecha dado desequilíbrio direcional de *ticks*.  
**Uso no Atlas**: `loaders.py` (criar funções `get_dollar_bars`, `get_imbalance_bars` e um `BarConfig` no `config.py`).

### CUSUM de retornos (detecção de evento)
- Acumula desvios até ±h·σ e dispara evento; h calibrado por *expected time between events*.  
**Atlas**: `factors.py` ou `loaders.py` (função `cusum_filter`).

### Triple-Barrier labeling (+ meta-labeling)
- Para cada evento i: barreira superior m⁺σ, inferior m⁻σ e **vertical** em Tᵥ.  
- Rótulo = qual barreira é tocada primeiro; *meta-labeling* aprende quando aceitar/rejeitar o sinal primário.  
**Atlas**: `risk_controls.py` (parâmetros de σ lookback e m±), `factors.py` (rótulos), `config.py` (`labeling:`).

### Pesos por sobreposição (sample weights)
- Penaliza exemplos com alta **concorrência** (muitos eventos ativos). *Uniqueness* promove exemplos “mais únicos”.  
**Atlas**: `purged_cv.py` (fornecer `sample_weight` ao estimador).

### Fractional Differencing (FracDiff)
- (1−L)^d xₜ = Σ wₖ xₜ₋ₖ, com wₖ = (−1)ᵏ·C(d,k). Truncar quando |wₖ| < τ.  
**Atlas**: `loaders.py` (`fracdiff(series, d, thresh)`), `config.py` (`fracdiff: {d: 0.4, thresh: 1e-4}`).

### Validação: Purged K-Fold + Embargo / CPCV
- **Purge**: remova do treino observações que se sobrepõem no tempo à janela do teste.  
- **Embargo**: remova janela E% após cada fold de teste para evitar vazamento por spillover.  
- **CPCV**: combina folds em múltiplos *train/test splits* para estimativa de risco mais estável.  
**Atlas**: já há `purged_cv.py`; exponha `embargo_pct` no `config.py`. `walk_forward.py` → *CPCV* opcional.

### Importância de features
- **MDA** (Permutation) com *purge+embargo*;  
- **SFI** (Single Feature) quando colinearidades são fortes.  
**Atlas**: `factors.py` (reportar `importances.json`), `kpis.py` (plot + estabilidade).

### Bet sizing (Kelly fracionário)
- sizeₜ = f·payout(p̂, q̂, b) com *floor/cap*, *step size* e *cooldown*.  
- Meta-labeling entrega probabilidade p̂ de acerto do classificador primário.  
**Atlas**: `position_sizing.py` (funções `kelly_fractional`, `meta_bet_sizer`).

### HRP / T-HRP
- Distância ρ→D = √(½(1−ρ)), *single-linkage* → *seriation* (quasi-diagonalization) → aloca por **IVP recursivo**.  
- T-HRP do Atlas: incorpore *TDA* para agrupar regimes e estabilizar clusters ao longo do tempo.  
**Atlas**: `hrp.py` (HRP base), `tda.py` (features topológicos) e `risk_controls.py` (limites por cluster).

### Backtest & Robustez
- **Walk-forward** com **Purged/CPCV**, **custos**, **slippage modelado**, **capacidade** (market impact), e *Combinatorial CV* para PSR estável.  
**Atlas**: `walk_forward.py`, `costs.py`, `capacity.py`, `execution.py`, `kpis.py`, `robustness.py`.

## 4) Boas práticas & *anti-patterns*
- Nunca usar **time-bars** puros para treinar; prefira barras informacionais.  
- Sempre aplicar **purge+embargo** em qualquer CV, inclusive em *feature importance*.  
- **FracDiff**: calibrar d via ADF/PP para *stationary-enough*; d muito alto mata memória, muito baixo mantém raiz-unitária.  
- **Label leakage**: cuidado com *future volatility* ao definir barreiras; use σ ex-ante (rolante).  
- **Score financeiro** > métricas genéricas; *maximize downside-aware utility* (e.g., Calmar, Omega).  
- **Capacidade**: valide *turnover* e *market impact*; *Kelly* deve ser **fracionário** e sujeito a *drawdown stops*.  
- **HRP**: verificar **cluster stability** (bootstraps) e impor **min/max weights**, **leverage** e **turnover**.

## 5) “Glue” para o `base.yaml` (exemplo mínimo)
```yaml
data:
  bars: {type: dollar_imbalance, target_usd: 1e6}
  fracdiff: {d: 0.45, thresh: 1e-4}
events:
  cusum: {vol_lookback: 50, h_sigma: 3.0}
labeling:
  method: triple_barrier
  vol_lookback: 50
  m_up: 2.0
  m_dn: 2.0
  t_vertical: 5d
validation:
  cv: purged_kfold
  n_splits: 5
  embargo_pct: 0.01
features:
  sets: [microstructure, tda, factors_core]
bet_sizing:
  method: kelly_fractional
  kelly_frac: 0.25
  size_cap: 0.05
portfolio:
  allocator: thrp
  risk_target_annual: 0.12
  weight_caps: {min: 0.0, max: 0.2}
```

## 6) Mapeamento direto para os arquivos do Atlas
- `loaders.py`: barras, fracdiff, CUSUM.  
- `factors.py`: features (microestrutura + TDA) e *meta-labels*.  
- `purged_cv.py`: Purged K-Fold + embargo; opcional CPCV.  
- `walk_forward.py`: encadear *train→tune→backtest* por janelas.  
- `position_sizing.py`: Kelly fracionário, *step*, *cooldown*, *caps*.  
- `hrp.py`: HRP; adaptar para T-HRP (clusters estáveis por regime/TDA).  
- `risk_controls.py`: barriers σ-based, leverage caps, kill-switches.  
- `costs.py`, `capacity.py`, `execution.py`: custos, impacto, *slippage*.  
- `kpis.py`: PSR, turnover, drawdowns, estabilidade de clusters.  
- `robustness.py`: bootstrap, *stress*, *subsampled CV*.

## 7) Prompts “prontos” para o seu especialista (GPT)
- “Dado `vol_lookback=50`, `m_up=m_dn=2`, `t_vertical=5d`, gere rótulos triple-barrier e pesos por *uniqueness*.”  
- “Faça Purged K-Fold com `embargo_pct=1%` e reporte PSR e CPCV-error.”  
- “Otimize `d` do FracDiff visando passar ADF (5%) e maximizar PSR fora da amostra.”  
- “Calcule MDA com purge+embargo e rankeie features; valide estabilidade.”  
- “Construa carteira T-HRP com `risk_target=12%`, `max_weight=20%`, `turnover_cap=10%/mês`.”

## 8) Checklists rápidos
- [ ] Barras ≠ time-bars na etapa de treino  
- [ ] σ é rolante ex-ante  
- [ ] Purge+Embargo em CV, FI e tuning  
- [ ] d calibrado (ADF pass)  
- [ ] Meta-labeling ligado quando houver *primary signal*  
- [ ] Kelly fracionário + caps + *drawdown stop*  
- [ ] THRP com estabilidade de clusters  
- [ ] Backtest com custos/impacto/capacidade
