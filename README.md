# t_hrp_v3.0

<<<<<<< Updated upstream
<<<<<<< Updated upstream
T-HRP v3.0 é uma base de pesquisa para estudar portfólios Hierarchical Risk Parity com pipeline completa: ingestão de dados, engenharia de sinais topológicos, alocação HRP, sizing, custos e validações avançadas (walk-forward, purged CV, robustez, capacidade). Os artefatos gerados ficam em `/reports`.

## Instalação
=======
T-HRP v3.0 e uma base de pesquisa para estudar portfolios Hierarchical Risk Parity com pipeline completa: ingestao de dados, engenharia de sinais topologicos, alocacao HRP, sizing, custos e validacoes avancadas (walk-forward, purged CV, robustez, capacidade). Os artefatos gerados ficam em `/reports`.

## Instalacao
>>>>>>> Stashed changes
=======
T-HRP v3.0 e uma base de pesquisa para estudar portfolios Hierarchical Risk Parity com pipeline completa: ingestao de dados, engenharia de sinais topologicos, alocacao HRP, sizing, custos e validacoes avancadas (walk-forward, purged CV, robustez, capacidade). Os artefatos gerados ficam em `/reports`.

## Instalacao
>>>>>>> Stashed changes

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
<<<<<<< Updated upstream
<<<<<<< Updated upstream
pip install -e .[dev]   # instala dependências + ferramentas (ruff/black/pytest)

pip install -e .   # instala dependências do pyproject
=======
.\.venv\Scripts\activate   # Windows
pip install -e .[dev]        # instala dependencias + ferramentas (ruff/black/pytest)
pip install -e .             # instala dependencias de runtime
>>>>>>> Stashed changes
=======
.\.venv\Scripts\activate   # Windows
pip install -e .[dev]        # instala dependencias + ferramentas (ruff/black/pytest)
pip install -e .             # instala dependencias de runtime
>>>>>>> Stashed changes
```

## Dados

<<<<<<< Updated upstream
<<<<<<< Updated upstream
Coloque arquivos CSV em `/data` com colunas `date`, `asset`, `close`, `volume`. O loader interpreta o nome do arquivo como ticker quando a coluna `asset` falta. O intervalo padrão usado na config vai de 2020-01-01 a 2022-12-31; ajuste conforme disponibilidade.

## Execução via CLI

Todos os fluxos são orquestrados pelo entrypoint:
=======
Coloque arquivos CSV em `/data` com colunas `date`, `asset`, `close`, `volume`. O loader usa o nome do arquivo como ticker quando a coluna `asset` nao existe. O intervalo padrao da config cobre 2020-01-01 a 2022-12-31; ajuste conforme disponibilidade.

## Execucao via CLI

Todos os fluxos sao orquestrados pelo entrypoint:
>>>>>>> Stashed changes
=======
Coloque arquivos CSV em `/data` com colunas `date`, `asset`, `close`, `volume`. O loader usa o nome do arquivo como ticker quando a coluna `asset` nao existe. O intervalo padrao da config cobre 2020-01-01 a 2022-12-31; ajuste conforme disponibilidade.

## Execucao via CLI

Todos os fluxos sao orquestrados pelo entrypoint:
>>>>>>> Stashed changes

```bash
python -m src.main --mode backtest --config configs/base.yaml
python -m src.main --mode walkforward --config configs/base.yaml
python -m src.main --mode tune --config configs/base.yaml
python -m src.main --mode robustness --config configs/base.yaml
python -m src.main --mode capacity --config configs/base.yaml
<<<<<<< Updated upstream
<<<<<<< Updated upstream
python -m src.main --mode report --config configs/base.yaml\npython -m src.reports.build_pdf --config configs/base.yaml --out reports/t_hrp_v3_report.pdf\n
```

* `backtest`: roda o HRP com custos/ATR/caps e grava equity curve.
* `walkforward`: 2 anos IS / 6 meses OOS com retreino.
* `tune`: purged K-fold com embargo (Sharpe OOS).
* `robustness`: heatmaps de sensibilidade, stress de custos, regimes TFI.
* `capacity`: variação do `participation_cap` e impacto em Sharpe/MaxDD.
* `report`: gera figuras/tabelas em `/reports` (ex.: `equity_curves_full.png`).

## Relatórios

Saídas principais ficam em `/reports` com timestamps:
=======
python -m src.main --mode report --config configs/base.yaml
python -m src.reports.build_pdf --config configs/base.yaml --out reports/t_hrp_v3_report.pdf
```

- `backtest`: roda o HRP com custos/ATR/caps e grava equity curve.
- `walkforward`: 2 anos in-sample / 6 meses out-of-sample com retreino.
- `tune`: purged K-fold com embargo (Sharpe OOS).
- `robustness`: heatmaps de sensibilidade, stress de custos, regimes TFI.
- `capacity`: variacao do `participation_cap` e impacto em Sharpe/MaxDD.
- `report`: gera figuras/tabelas em `/reports` (ex.: `equity_curves_full.png`).

## Relatorios

Saidas principais ficam em `/reports` com timestamps:
>>>>>>> Stashed changes
=======
python -m src.main --mode report --config configs/base.yaml
python -m src.reports.build_pdf --config configs/base.yaml --out reports/t_hrp_v3_report.pdf
```

- `backtest`: roda o HRP com custos/ATR/caps e grava equity curve.
- `walkforward`: 2 anos in-sample / 6 meses out-of-sample com retreino.
- `tune`: purged K-fold com embargo (Sharpe OOS).
- `robustness`: heatmaps de sensibilidade, stress de custos, regimes TFI.
- `capacity`: variacao do `participation_cap` e impacto em Sharpe/MaxDD.
- `report`: gera figuras/tabelas em `/reports` (ex.: `equity_curves_full.png`).

## Relatorios

Saidas principais ficam em `/reports` com timestamps:
>>>>>>> Stashed changes

- `equity_curve_*.csv` / `equity_curves_*.png`
- `tuning_results.csv`
- `heatmap_sharpe_*.png`, `heatmap_vol_*.png`
- `stress_costs_*.csv`
- `capacity_curve_*.csv` e `capacity_curve_*.png`
- `regime_kpis.csv`

<<<<<<< Updated upstream
<<<<<<< Updated upstream
Adicionalmente, há um notebook esqueleto em `scripts/sanity_notebook.ipynb` para inspeções manuais de distribuição de scores e rank IC de curto prazo.

## Limitações & Próximos Passos

- Dados de exemplo são sintéticos; adapte o loader (`dataio/loaders.py`) para feeds reais.
- Modelagem de slippage e custos é simples (linear em participação). Avaliar modelos não-lineares.
- Falta persistência de sinais/pesos para consumo por OMS.
- Validar e expandir métricas (ex.: rolling hit rate, expected shortfall).
- Adicionar suporte a execução distribuída dos grids de tuning/robustez.
=======
Ha um notebook esqueleto em `scripts/sanity_notebook.ipynb` para inspecoes manuais de distribuicoes de scores e rank IC de curto prazo.

## Limitacoes e Proximos Passos

- Dados de exemplo sao sinteticos; adapte o loader (`dataio/loaders.py`) para feeds reais.
- Modelagem de slippage e custos e simples (linear em participacao). Avaliar modelos nao lineares.
- Falta persistencia de sinais/pesos para consumo por OMS.
- Validar e expandir metricas (ex.: rolling hit rate, expected shortfall).
- Adicionar suporte a execucao distribuida dos grids de tuning/robustez.
>>>>>>> Stashed changes

## Qualidade e Testes

```bash
ruff check src tests
black src tests scripts
pytest -q
```
<<<<<<< Updated upstream

=======
Ha um notebook esqueleto em `scripts/sanity_notebook.ipynb` para inspecoes manuais de distribuicoes de scores e rank IC de curto prazo.

## Limitacoes e Proximos Passos

- Dados de exemplo sao sinteticos; adapte o loader (`dataio/loaders.py`) para feeds reais.
- Modelagem de slippage e custos e simples (linear em participacao). Avaliar modelos nao lineares.
- Falta persistencia de sinais/pesos para consumo por OMS.
- Validar e expandir metricas (ex.: rolling hit rate, expected shortfall).
- Adicionar suporte a execucao distribuida dos grids de tuning/robustez.

## Qualidade e Testes

```bash
ruff check src tests
black src tests scripts
pytest -q
```
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
