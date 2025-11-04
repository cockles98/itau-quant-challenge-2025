# Atlas, the market cartographer

## Executive Summary
Atlas, the market cartographer is a full research and execution framework for regime-aware hierarchical risk parity. The platform combines persistent homology turbulence (PH), Mapper-based topology, classical factor overlays, and machine-learning meta-models (Ridge / ElasticNet) to produce robust allocations for a long-only Brazilian equity universe. The complete study spans **14 Sep 2017 to 06 Oct 2025** with the following headline results:

-  **Walk-forward (504d IS / 126d OOS, rolling, Selic excess):** Sharpe 1.18, annual return 31.6%, annual vol 16.5%, max drawdown -19.9%.
- **Validity checks:** Probabilistic Sharpe ratio ~1.0 with deflated p-value 11.6% (~88.4% confidence that the excess over the Selic rate is not noise).

Each research component is modular, traceable, and designed for professional auditability: configs are YAML-based, artifacts are persisted to `artifacts/` and `reports/`, and every run can be reproduced with a single CLI command.

## Performance Highlights
- **Equity curves:** comparison between rolling walk-forward and Ibovespa benchmark.
- **Summary KPIs:** side-by-side statistics for the full backtest, walk-forward evaluation, and benchmark.
- **Walk-forward detail:** KPIs for each out-of-sample window, exposing regime dispersion.
- **Benchmark overlay:** walk-forward results versus the Ibovespa, highlighting excess return, tracking error, and style biases.

Equity curves:

OBS: The strategy starts generating out-of-sample results on **30 March 2020**, because the data between **14 Sep 2017**, and that date are only used to train the first block of the walk-forward.

![Equity Curve Comparison - Backtest](data/readme_assets/equity_curve_comparison_v2.png)

Summary KPIs:

![Key Performance Metrics](data/readme_assets/kpi_table_v2.png)

KPIs for each out-of-sample window from walk-forward:

![Walk-forward Window Metrics](data/readme_assets/walkforward_window_metrics_v2.png)

Walk-forward excess performance over the Ibovespa (Walk-forward / Ibovespa):

![Curva de Excesso Acumulado](data/readme_assets/benchmark_curve.png)

KPIs of the walk-forward allocation against the Ibovespa (alpha, beta, excess return, tracking error, information ratio, and correlation):

![Benchmark KPIs](data/readme_assets/benchmark_kpis.png)

## Robustness & Validation
- **Risk parameter plateau:** Sensitivity heatmaps point to a stable region around `risk.target_vol` 0.12-0.14 e `risk.vol_mult` 1.4-1.6. Outside that band Sharpe decays quickly, signalling the final configuration is not a narrow optimum.
- **Alternative risk profile:** A leaner setup (`target_vol=0.14`, `vol_mult=1.4`) entrega Sharpe 0.87, CAGR 25.1%, max drawdown -16.5%, oferecendo alternativa mais conservadora com ~6 p.p. a menos de retorno anual.
- **Statistical sanity checks:** Walk-forward base mantem PSR ~1.0 e p-valor deflacionado 11.6% (>88% de confianca). A variante conservadora retorna Sharpe 0.87 com p-valor deflacionado 32.6% (~67% de confianca).

## Data and Time Horizon
- **Universe:** Constituents of the Ibovespa index for each quadrimester (aligned with the official rebalancing schedule). From that universe we trade the top 20 names by ADV, subject to price (> BRL 5), age (> 20 business days), and hysteresis (4 rebalances) filters.
- **Raw inputs:** Local CSV files in `data/` containing `date`, `asset`, `close`, and `volume` columns. Each file already reflects the Ibovespa constituent list for its quadrimester. The loader normalises dates to business frequency and computes ADV, ATR, and other derived metrics.
- **Risk-free:** Taxa Selic diaria (`data/selic/taxa_selic_apurada.csv`) alimenta os calculos de retorno em excesso (Sharpe/Sortino).
- **Study window:** 2017-09-04 through 2025-10-06 (configurable via `configs/base.yaml`). Walk-forward windows operate on 504 business days in-sample and 126 days out-of-sample.

## Pipeline Overview
1. **Data ingestion (`src/dataio`):** loads the quadrimester-specific Ibovespa panels, caches parquet snapshots, and applies hysteresis-based universe selection.
2. **Feature engineering (`src/features`):**
   - Persistent homology regime index via `compute_ph_regime_index` (window=30, z-score lookback=250).
   - Mapper topology (`RegimeAwareMapper`) for peripherality tilt and HRP seriation guidance.
   - Momentum (12-1), quality, and carry proxies.
   - **Meta-blend machine learning overlay** (`src/models/meta_blend.py`): Ridge or ElasticNet regressors learn an optimal mix of factor scores (with optional regime and interaction terms), providing data-driven weights that adapt to changing market regimes.
3. **Portfolio construction (`src/backtest`):** HRP base weights with optional expected-return tilt (including ML-derived scores), periphery bias, and regime-aware re-scaling of target volatility, gross exposure, and participation caps.
4. **Execution modelling:** Trading costs (fees, non-linear slippage), ATR-based position sizing, turnover caps, and cluster-level risk guards.
5. **Risk management:** Kill-switch (rolling MDD and realised vol), cooldown rehits, regime-driven limits, capacity tracking (`meta/regime_controls_*`).
6. **Validation (`src/validation`):** walk-forward evaluation, purged CV, Mapper/PH robustness heatmaps, risk tuning (grid search), and capacity curves.
7. **Reporting (`src/reports`, `notebooks/`):** CSV/PNG artifacts plus a consolidated PDF summarising equity curves, KPIs, heatmaps, and regime diagnostics.

## Persistent Homology Regime Index
Atlas relies on persistent homology (PH) to gauge turbulence in the Ibovespa universe. The implementation (`src/features/regime/ph_regime.py`) slides a 30-day window over cross-sectional returns, builds Vietoris-Rips diagrams, and converts persistence landscapes into a smoothed z-score regime series. Configuration knobs (`tda_ph.window`, `tda_ph.alert_sigma`, `tda_ph.riskoff_sigma`) let the strategy tighten or loosen exposure as stress levels rise. The resulting regime curve drives target-vol scaling, gross exposure gating, kill-switch hysteresis, and the adaptive capacity controls saved under `meta/regime_controls_*`.

## Mapper Topology and Peripherality
Mapper topology (`src/features/tda/mapper.py`) complements PH by projecting assets through a lens (default PCA+UMAP), covering the space with overlapping cubes, and clustering via DBSCAN. Each rebalance snapshot produces metrics such as number of connected components, average degree, and node-size Gini, cached to `reports/mapper_metrics_*.csv`. Mapper centrality feeds two core levers: (i) HRP seriation (`src/portfolio/hrp_topo.py`) uses the topology to stabilise covariance splits, and (ii) peripherality bias (`portfolio.weighting.apply_periphery_bias`) tilts allocations toward safer cores or away from riskier fringes according to `portfolio.periphery_bias_lambda` and `factors.delta`.

## Hierarchical Risk Parity Core
At the heart of Atlas sits a hierarchical risk parity engine (`src/backtest/engine.py`) that marries HRP clustering with regime-aware risk controls. Covariance matrices are computed via `portfolio.rolling_cov`, optionally cached and shrunk, then reordered using Mapper-enhanced seriation before recursive bisection allocates risk. Expected-return tilts (`portfolio.expected_sharpe_tilt`) and periphery adjustments apply on top, while ATR-based sizing and turnover/participation caps ensure execution realism. The engine records full telemetry (weights, trades, regime stats) and supports both single-run backtests and rolling walk-forward evaluations.

## Risk Management and Kill-Switch Framework
Risk governance in Atlas is centralised inside `src/backtest/engine.py` and `src/risk`. A rolling MDD/volatility kill-switch (`risk.guards`) halts trading during extreme drawdowns and enforces cooldown hysteresis before re-entry. Participation, cluster, and turnover caps interpolate with the PH regime to expand or shrink exposure, while ATR-based sizing (`risk.position_sizing`) keeps position risk aligned with targets. Capacity telemetry (stored under `meta/regime_controls_*`) tracks utilisation and binding events, feeding stress analyses and dashboards. These layers ensure HRP allocations remain responsive without breaching liquidity or risk budgets.

## Machine Learning Meta-Blend
Atlas employs supervised learning to enhance the factor overlay. The module (`src/models/meta_blend.py`) prepares features from momentum, quality, carry, PH regime, and interaction terms; applies purged K-fold CV with embargo; and fits Ridge or ElasticNet regressors according to the `factors.meta_blend` configuration. Users can tune grids for `alpha`, `l1_ratio`, lookback horizon, rolling window, and caching options, or run scenario batches via `scripts/run_meta_blend_scenarios.py`. When enabled, the learned mix replaces static factor weights with regime-aware combinations that adapt to current market states while honouring the HRP structure and risk controls.

## Repository Structure
```
configs/          YAML configurations (base, hrp_only, tda_only, meta_blend*)
src/
  dataio/         Loaders and config validation
  features/       Mapper, PH turbulence, factor signals
  backtest/       Deterministic engine and execution scaffolding
  validation/     Walk-forward, risk tuning, robustness utilities
  risk/           Position sizing, kill-switches, guards
  reports/        Table/figure generation and PDF builder
scripts/          Batch experiments (meta-blend grids, strategy comparison, etc.)
notebooks/        End-to-end and reporting notebooks
tests/            Pytest suite covering factors, mapper, risk tuning, regimes
artifacts/, reports/, meta/   Generated outputs (equity curves, CSVs, dashboards)
```

## Installation and Environment
```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell / Windows
# or source .venv/bin/activate  # Linux / macOS

pip install -e ".[dev]"         # runtime + ruff + pytest + black
```
Key dependencies include `pandas`, `numpy`, `networkx`, `umap-learn`, `kmapper`, `giotto-tda`, and `riskfolio-lib`. A Python >=3.10 interpreter is required.

## Configuration
Primary settings live in `configs/base.yaml`:
- `portfolio`: base method (`hrp`, `hrp_only`, `tda_only`), periphery bias lambda, expected-return tilt knobs.
- `tda_ph`: PH turbulence parameters (`window`, `smooth span`, `z-score lookback`, `alert/risk-off sigmas`).
- `mapper`: lens selection, resolution (`n_cubes`, `overlap`), epsilon quantile, min cluster size.
- `factors`: factor list, blending weights (`alpha`, `beta`, `gamma`, `delta`), meta-blend model.
- `risk`: target volatility, regime scaling bounds, kill-switch lookbacks, participation/turnover caps.
- `validation`: grids for PH, Mapper, factor weights, robustness sweeps, and risk stress testing.
Alternative configs (`hrp_only.yaml`, `tda_only.yaml`, `meta_blend*.yaml`) inherit the same structure.

## Reproducing Results
### Core runs
```bash
python -m src.main --mode backtest    --config configs/base.yaml
python -m src.main --mode walkforward --config configs/base.yaml
python -m src.main --mode tune        --config configs/base.yaml
python -m src.main --mode robustness  --config configs/base.yaml
python -m src.main --mode capacity    --config configs/base.yaml
python -m src.reports.build_pdf --config configs/base.yaml --out reports/atlas_report.pdf
```

### Risk tuning and visualisations
```bash
python -m scripts.run_meta_blend_scenarios --config configs/meta_blend.yaml
python -m scripts.run_strategy_comparison --base-config configs/base.yaml \
       --hrp-config configs/hrp_only.yaml --tda-config configs/tda_only.yaml
python -m scripts.run_participation_cap_grid --config configs/base.yaml
python -m scripts.run_scale_cap_combo_grid  --config configs/base.yaml
python -m scripts.run_vol_grid_v2           --config configs/base.yaml
python -m scripts.run_kill_switch_grid      --config configs/base.yaml
python -m scripts.run_ph_threshold_backtest --config configs/base.yaml
```
All scripts respect `paths.artifacts` and `paths.reports` overrides; use `isolate_artifacts=True` flags (where available) to keep scenario caches segregated.

## Testing and Quality Assurance
The full automated suite can be executed with:
```bash
python -m pytest -q
```
Tests cover mapper metrics, PH regime computation, HRP seriation with and without periphery bias, risk tuning grids, and walk-forward slicing logic. Linting and formatting are available via:
```bash
ruff check src tests
black src tests scripts
```

## Generated Artifacts
- `reports/equity_curve.csv`, `reports/walkforward_equity.csv`: equity series per mode.
- `reports/mapper_metrics_*.csv`, `meta/regime_controls_*.csv`: topology and regime telemetry.
- `reports/heatmap_*.png`, `reports/stress_costs_*.csv`: robustness and stress analyses.
- `reports/atlas_report.pdf`: consolidated document with KPIs, graphs, tables, and tuning summaries.
- `artifacts/cache/`: cached factor, covariance, Mapper, and regime computations to speed up reruns.

## Key Insights
- PH turbulence acts as the primary regime filter controlling gross exposure, target volatility, and capacity caps. Alert sigma 0.58 and risk-off sigma 1.8 were empirically tuned.
- Mapper-based peripherality provides a meaningful overlay: periphery bias lambda of 1.0 improves risk-adjusted returns while preserving diversification (max cluster regime caps between 7.5% and 9%).
- Risk tuning modules allow rapid exploration of participation caps, target vol ranges, and kill-switch settings without re-running full notebooks.
- Machine-learning meta-blend (Ridge/ElasticNet) adapts factor weights to current regimes, consistently improving out-of-sample Sharpe in walk-forward analyses.

## Operational Notes
- Universe selection uses hysteresis to avoid excessive churn; cached universes live under `artifacts/cache/universe`.
- Run-time caches (factors, Mapper, covariance) are keyed by start/end dates, sample size, and config hashes. Remove corresponding subdirectories if a clean rebuild is needed.
- For production deployment integrate `run_backtest` / `run_walk_forward` with a scheduler, ensuring the `data/` directory is updated with the latest end-of-day files.

## Roadmap
- Integrate additional macroeconomic features (rates, FX) as Mapper lenses to enhance regime discrimination.
- Extend meta-blend to incorporate PH regime features and forward-looking risk metrics in the learning set.
- Add unit tests for `scripts/` entrypoints and expand coverage for expected-return tilting edge cases.
- Containerise the environment for consistent cloud execution and CI automation.
