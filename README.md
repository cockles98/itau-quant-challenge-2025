# T-HRP v3.0

T-HRP v3.0 e uma base de pesquisa para montar carteiras long-only usando Topological-HRP Plus. O projeto combina mapeamento topologico de regimes, overlay de fatores classicos e gestao de risco ativa com validacao rigorosa e instrumentacao completa.

## Destaques
- Pipeline end-to-end cobrindo ingestao, sinais, alocacao HRP, sizing, execucao e relatorios.
- Sinais topologicos com KeplerMapper/TFI integrados a fatores momentum, quality e carry, com meta blend regularizado opcional.
- Gestao de risco ativa com alvo de volatilidade, sizing por ATR, caps de participacao e kill switch parametrizavel.
- Validacoes walk-forward, purged K-fold com embargo, analises de robustez e curva de capacidade nativas.
- Scripts e relatorios consolidados (PDF) prontos para entrega e defesa da tese quantitativa.

## Estrutura do Projeto
- `configs/`: YAMLs com parametros padrao (`configs/base.yaml`).
- `src/dataio/`: carregamento de paineis OHLCV (CSV) e selecao de universo com histerese.
- `src/features/`: sinais topologicos (TFI) e fatores classicos auxiliares.
- `src/portfolio/`: HRP com seriation topologica e covariancia rolling.
- `src/risk/`: funcoes de sizing (ATR, vol targeting) e controles de risco.
- `src/backtest/`: motor deterministico com custos, execucao e cache de fatores/covariancias.
- `src/validation/`: walk-forward, purged CV, heatmaps, estresse de custos e capacidade.
- `src/reports/`: geracao de tabelas, figuras e PDF final.
- `scripts/`: utilitarios para grids, sensibilidades e exportacao de mapas TDA.
- `notebooks/`: `end_to_end_pipeline.ipynb` detalha o fluxo completo.
- `tests/`: suite pytest cobrindo covariancia, TDA, HRP, validacao e funcoes auxiliares.

## Instalacao
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# ou
.\.venv\Scripts\activate     # Windows

pip install -e .[dev]        # dependencias + ferramentas (ruff/black/pytest)
pip install -e .             # apenas runtime, se preferir
```
Observacao: o projeto requer Python >= 3.10 e usa pandas, riskfolio-lib, scikit-learn e networkx.

## Preparacao dos Dados
- Coloque CSVs em `data/` com colunas `date`, `asset` (opcional), `close`, `volume`.
- Datas sao normalizadas para frequencia diaria; mantenha timezone neutro.
- Quando `asset` nao existe, o nome do arquivo vira ticker.
- Ajuste `dates.start`/`dates.end` em `configs/base.yaml` conforme a sua amostra.
- Dados sao cacheados em Parquet em `artifacts/cache` para recargas mais rapidas.

## Como Rodar
```bash
python -m src.main --mode backtest --config configs/base.yaml
python -m src.main --mode walkforward --config configs/base.yaml
python -m src.main --mode tune --config configs/base.yaml
python -m src.main --mode robustness --config configs/base.yaml
python -m src.main --mode capacity --config configs/base.yaml
python -m src.main --mode report --config configs/base.yaml
python -m src.reports.build_pdf --config configs/base.yaml --out reports/t_hrp_v3_report.pdf
# utilidades extras acionadas por flags
python -m src.main --mode backtest --config configs/base.yaml --export-tda-maps
python -m src.main --mode backtest --config configs/base.yaml --tda-sensitivity
python -m src.main --mode backtest --config configs/base.yaml --ph-threshold-bt
```
- `backtest`: HRP com custos, ATR sizing e caps; salva `reports/equity_curve.csv`.
- `walkforward`: janelas 2 anos IS / 6 meses OOS com retreino completo.
- `tune`: purged K-fold com embargo para hiperparametros TDA e pesos de fatores.
- `robustness`: heatmaps de sensibilidades, estresse de custos e KPI por regime TFI.
- `capacity`: curva Sharpe/MDD variando `participation_cap`.
- `report`: gera figuras/tabelas do rebalance atual.
- `build_pdf`: consolida artefatos em PDF final.
- `--export-tda-maps`: exporta PNG/JSON/metrics do Mapper para `artifacts/tda/maps`.
- `--tda-sensitivity`: roda a malha `n_cubes × overlap` e grava CSV/PNG/JSON em `reports/tda_sensitivity.*`.
- `--ph-threshold-bt`: avalia o filtro de turbulência PH e salva `reports/ph_threshold_backtest.*`.

## Configuracoes Principais (`configs/base.yaml`)
- `tda`: delay, dimensao, numero de cubos, overlap, epsilon adaptativo, janela e smoothing do TFI.
- `mapper`: lente baseline (`pca_umap`), resolucao (`n_cubes`, `overlap`), eps adaptativo e min_cluster_size.
- `factors`: fatores base, pesos (`alpha`, `beta`, `gamma`, `delta`) e modulo `meta_blend` (Elastic-Net supervisionado por IC).
- `windows`: janelas de volatilidade (60d) e ATR (60d) utilizadas no sizing.
- `costs`: comissao base (5 bps) e curva de slippage nao linear (`k`, `max_bps`).
- `universe`: top 20 por ADV com filtros de preco, idade e histerese de 4 rebalanceamentos.
- `risk`: alvo de vol (10%), escala dinamica (`regime_scale_low/high`), caps (`participation_cap`, `turnover_cap`), kill switch via MDD/vol e cooldown.
- `validation`: grids de tuning/robustez, multiplicadores de stress e participacao maxima testada.

## Metodologia T-HRP v3.0
1. Universo: top 20 em liquidez com filtros de ADV, preco, idade e histerese de 2 rebalanceamentos.
2. Sinais: TFI/KeplerMapper identificam regimes e sao combinados com momentum, quality/carry e meta blend regularizado.
3. Carteira: HRP topologico gera pesos base e distribui budget por cluster ajustado pelo score combinado.
4. Sizing: normalizacao por ATR, caps por ativo/cluster e escala ate o alvo de volatilidade.
5. Controles: turnover maximo 25%, participation rate ate 5% do ADV, kill switch de MDD/vol e exposicao long-only.
6. Validacao: walk-forward, purged CV com embargo, heatmaps de robustez, estresse de custos e curva de capacidade.
7. Relatorios: equity curves, KPIs, regime tables, tuning grids, heatmaps e PDF consolidado.

## Saidas Geradas em `reports/`
- `equity_curve*.csv/.png` e `walkforward_equity.csv`.
- `tuning_results.csv` com grid search completo.
- `heatmap_*.png` para sensibilidades de TDA e fatores.
- `stress_costs_*.csv` com PnL sob multiplicadores de custos.
- `capacity_curve_*.csv/.png` para limite de participacao.
- `regime_kpis.csv` com desempenho por faixa de TFI e `tda_dashboard.png` (série PH z-score, `n_components`, `target_vol_eff` vs `gross`).
- `t_hrp_v3_report.pdf` via `build_pdf`.

## Testes e Qualidade
```bash
ruff check src tests
black src tests scripts
pytest -q
```
Os testes cobrem TDA, HRP, covariancia rolling, validacao, metricas e utilitarios. Execute-os antes de subir mudancas.

## Scripts Auxiliares
- `scripts/run_meta_blend_scenarios.py`: varre configuracoes de Ridge/Elastic-Net e grava KPIs.
- `scripts/run_tda_sensitivity.py`: gera heatmaps customizados para TDA/TFI.
- `scripts/run_ph_threshold_backtest.py`: testa thresholds do índice de turbulência (mesma função de `--ph-threshold-bt`).
- `scripts/run_kill_switch_grid.py`: avalia thresholds de MDD e volatilidade.
- `scripts/run_participation_cap_grid.py`: compara limites de participacao em ADV.
- `scripts/run_scale_cap_combo_grid.py`: combina ajustes de vol e participacao.
- `scripts/export_tda_maps.py`: exporta grafos Mapper/TFI para analise externa (equivalente ao flag `--export-tda-maps`).

## Limitacoes e Proximos Passos
- Substituir os dados sinteticos por feeds reais (ajuste `src/dataio/loaders.py`).
- Afinar a modelagem de slippage para mercados especificos (impacto nao linear mais realista).
- Persistir sinais/pesos para integracao com OMS e execucao continua.
- Adicionar metricas adicionais de risco (expected shortfall, hit-rate rolling) e dashboards dedicados.
- Distribuir grids de tuning/robustez em cluster (joblib backend ou Ray).
- Expandir o overlay de fatores e testar novas combinacoes com purged CV.
