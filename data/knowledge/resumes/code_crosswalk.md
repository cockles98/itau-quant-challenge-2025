
# Atlas Code Crosswalk (código ↔ fontes)

tags: [crosswalk, mapeamento, referencias, atlas]

## tda.py (Topological Regime / TFI)
- **Tierny (slides) + TTK docs + KeplerMapper docs**
  - Parâmetros: `delay`, `dim`, `n_cubes`, `overlap`, `epsilon`, `min_samples`, `window`
  - Decisões: lente (1ª coord.), cobertura sobreposta, DBSCAN por intervalo, arestas entre cubos adjacentes
  - Saídas: `tfi_score` normalizado [0,1], `tfi_stats` para diagnósticos

## hrp.py (Hierarchical Risk Parity)
- **HRP (SSRN)** + **APM**
  - `rolling_cov(window)`, `topo_seriation_from_graph(cov, graph)`, `hrp_weights_from_order(order)`
  - Distância: \(D = \sqrt{0.5(1-\rho)}\); alocação por variância de cluster (IVP recursivo)

## factors.py (momentum/quality/mix)
- **Ilmanen** (fatores), **ML4AM** (redes/HRP), **AFML** (normalizações)
  - `momentum_12_1`, `quality_proxy`, `mix_scores(regime, momentum, quality, αβγ)`
  - Dials: `softmax_T`, `REGIME_GAIN`, `REGIME_MODE` (tanh/power/linear)

## engine.py (colagem do pipeline)
- **APM** (IC/IR), **AFML** (validação), **Ilmanen** (regimes), **Hull** (limiares de risco)
  - Vol target, caps por ativo/cluster, turnover cap, kill-switch com cooldown/histerese

## risk_controls.py
- **Hull** (gestão de risco), **Almgren–Chriss** (impacto/capacidade)
  - `apply_caps`, `apply_turnover_cap`, `kill_switch(MDD/vol)`

## position_sizing.py
- **AFML** (bet sizing) + práticas de target vol/ATR
  - `atr`, `atr_risk_normalize`, `scale_to_vol`

## walk_forward.py / purged_cv.py
- **AFML** (Purged K-Fold + Embargo, CPCV)
  - Janelas 2y/6m OOS; validações sem vazamento

## capacity.py / execution.py / costs.py
- **Almgren–Chriss** (fronteira risco-impacto), notas de square-root law
  - `slippage_cost(k, max_bps)`, `commission_cost`, `capacity_curve` com `cap_bind_rate`

## robustness.py / kpis.py
- **AFML** (PSR, estabilidade), **APM** (IR), métricas padrão (Sharpe/Sortino/Calmar)
  - Heatmaps de sensibilidade (TDA e alphas), stress de custos e regimes
