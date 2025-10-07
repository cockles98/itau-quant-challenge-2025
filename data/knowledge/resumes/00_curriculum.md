
# Atlas (T-HRP) — Curriculum de Ingestão para o Especialista

tags: [curriculum, leitura, atlas, thrp, tda, hrp, validacao, execucao]

## Objetivo
Destilar as referências essenciais em **unidades curtas** (≤1200 palavras cada) que alimentam diretamente o código do projeto **Quant Challenge / Atlas**.

## Fontes “Core” (8)
1. AFML — Advances in Financial Machine Learning (López de Prado, 2018)
2. ML4AM — Machine Learning for Asset Managers (López de Prado)
3. HRP (paper SSRN) — Hierarchical Risk Parity
4. TDA — Tierny (slides) + TTK (docs) + KeplerMapper (docs)
5. APM — Active Portfolio Management (Grinold & Kahn)
6. Expected Returns (Ilmanen)
7. Almgren–Chriss (Optimal Execution) + notas de impacto
8. Hull — Risk Management & Financial Institutions

## Opcionais (quando houver espaço)
- ESL (Hastie/Tibshirani/Friedman) — CV/regularização
- Python for Finance (Hilpisch) — práticas pandas/numpy
- Unificação HRP↔MinVar (arXiv)

---

## Trilha de 6 semanas (curta e direta)

### Semana 1 — Validação correta e dados
- **AFML**: Purged K-Fold, Embargo, CPCV; barras informacionais; Triple-Barrier.
- **Tarefas**: habilitar `embargo_pct` no `purged_cv.py`; esqueleto `fracdiff` e `cusum_filter` no `loaders.py`.
- **Teste de aceitação**: roda CV purgada sem vazamento; relatório de *uniqueness*.

### Semana 2 — TDA/Mapper e regime
- **Tierny + TTK + KeplerMapper**: lens→cover→cluster→grafo; ε adaptativo; escolha de `n_cubes` & `overlap`.
- **Tarefas**: revisar `tda.py` (ε adaptativo, DBSCAN por cube, métricas do grafo); salvar `tfi_stats`.
- **Teste**: *heatmap* de sensibilidade (`robustness.py`) mostra variação não nula de TFI.

### Semana 3 — HRP e seriação topológica
- **HRP (SSRN)** + **APM** capítulos de alocação ativa.
- **Tarefas**: validar `topo_seriation_from_graph` e `hrp_weights_from_order`; checar *cluster variance*.
- **Teste**: pesos somam 1, variância por cluster decrescente na bisseção; comparação 1/N vs HRP.

### Semana 4 — Fatores, mix e sizing
- **Ilmanen** (fatores/regimes) + **ML4AM** (clustering/networks) + AFML (bet sizing).
- **Tarefas**: revisar `mix_scores` (α>β≥γ, regime_gain/mode), `atr_risk_normalize`, `scale_to_vol`.
- **Teste**: contribuição de risco mais homogênea (ATR) e vol realizada ≈ alvo.

### Semana 5 — Execução e Capacidade
- **Almgren–Chriss** (impacto×risco) + notas de *square-root law*.
- **Tarefas**: confirmar `participation_cap` e `slippage_cost(k, max_bps)`; produzir `capacity_curve`.
- **Teste**: *capacity curve* com Sharpe vs cap e anotações de `cap_bind_rate`.

### Semana 6 — Risco, kill switch e robustez
- **Hull** (VaR/ES, drawdown/vol) + AFML (PSR/estabilidade).
- **Tarefas**: `kill_switch` (MDD/vol, cooldown, histerese) e *stress* de custos/parametria.
- **Teste**: regime de desarme/rearme dispara corretamente; `stress_costs.csv` gerado.

---

## Diretrizes de escrita das notas
- **Máx. 1200 palavras**, com os cabeçalhos padrão: *Tese*, *Fórmulas*, *Dials*, *Pitfalls*, *Mapa para o código*.
- **Zero trechos longos copiados**; escrever em suas palavras.
- **Inclua tags** no topo: `tags: [TDA, epsilon, overlap, HRP, turnover, capacity]`.

## Entrega mínima
- `afml_core.md` (pronto)
- `HRP_SSRN_notes.md` (exemplo abaixo criado)
- `code_crosswalk.md` (feito neste pacote)
