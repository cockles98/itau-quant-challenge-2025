# T-HRP v3.0 Quantitative Portfolio Research

## Executive Summary
T-HRP v3.0 is a full research and execution framework for regime-aware hierarchical risk parity. The platform combines persistent homology turbulence (PH), Mapper-based topology, classical factor overlays, and disciplined risk management to produce robust allocations for a long-only Brazilian equity universe. The complete study spans **04 Sep 2017 to 06 Oct 2025** with the following headline results:

- **Walk-forward (504d IS / 126d OOS, rolling):** Sharpe 1.69, annual return 28.3%, annual vol 16.7%, max drawdown -20.4%.
- **Full backtest (same universe, single pass):** Sharpe 1.77, annual return 29.2%, annual vol 16.5%, max drawdown -20.2%.

Each research component is modular, traceable, and designed for professional auditability: configs are YAML-based, artifacts are persisted to `artifacts/` and `reports/`, and every run can be reproduced with a single CLI command.

## Data and Time Horizon
- **Universe:** Top 20 B3 equities by ADV, subject to price (> BRL 5), age (> 20 business days), and hysteresis (4 rebalances) filters.
- **Raw inputs:** Local CSV files in `data/` containing `date`, `asset`, `close`, and `volume` columns. Loader normalises dates to business frequency and computes ADV, ATR, and other derived metrics.
- **Study window:** 2017-09-04 through 2025-10-06 (configurable via `configs/base.yaml`). Walk-forward windows operate on 504 business days in-sample and 126 days out-of-sample.

## Pipeline Overview
1. **Data ingestion (`src/dataio`):** loads OHLCV panels, caches parquet snapshots, and applies hysteresis-based universe selection.
2. **Feature engineering (`src/features`):**
   - Persistent homology regime index via `compute_ph_regime_index` (window=30, z-score lookback=250).
   - Mapper topology (`RegimeAwareMapper`) for peripherality tilt and HRP seriation guidance.
   - Momentum (12-1), quality, and carry proxies; optional ElasticNet/Ridge meta-blend.
3. **Portfolio construction (`src/backtest`):** HRP base weights with optional expected-return tilt, periphery bias, and regime-aware re-scaling of target volatility, gross exposure, and participation caps.
4. **Execution modelling:** Trading costs (fees, non-linear slippage), ATR-based position sizing, turnover caps, and cluster-level risk guards.
5. **Risk management:** Kill-switch (rolling MDD and realised vol), cooldown rehits, regime-driven limits, capacity tracking (`meta/regime_controls_*`).
6. **Validation (`src/validation`):** walk-forward evaluation, purged CV, Mapper/PH robustness heatmaps, risk tuning (grid search), and capacity curves.
7. **Reporting (`src/reports`, `notebooks/`):** CSV/PNG artifacts plus a consolidated PDF summarising equity curves, KPIs, heatmaps, and regime diagnostics.

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
- `tda_ph`: PH turbulence parameters (window, smooth span, z-score lookback, alert/risk-off sigmas).
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
python -m src.reports.build_pdf --config configs/base.yaml --out reports/t_hrp_v3_report.pdf
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
- `reports/t_hrp_v3_report.pdf`: consolidated document with KPIs, graphs, tables, and tuning summaries.
- `artifacts/cache/`: cached factor, covariance, Mapper, and regime computations to speed up reruns.

## Key Insights
- PH turbulence acts as the primary regime filter controlling gross exposure, target volatility, and capacity caps. Alert sigma 0.58 and risk-off sigma 1.8 were empirically tuned.
- Mapper-based peripherality provides a meaningful overlay: periphery bias lambda of 1.0 improves risk-adjusted returns while preserving diversification (max cluster regime caps between 7.5% and 9%).
- Walk-forward Sharpe 1.69 demonstrates stability across rolling windows, outperforming static HRP-only baselines.
- Risk tuning modules allow rapid exploration of participation caps, target vol ranges, and kill-switch settings without re-running full notebooks.

## Operational Notes
- Universe selection uses hysteresis to avoid excessive churn; cached universes live under `artifacts/cache/universe`.
- Run-time caches (factors, Mapper, covariance) are keyed by start/end dates, sample size, and config hashes. Remove corresponding subdirectories if a clean rebuild is needed.
- For production deployment integrate `run_backtest` / `run_walk_forward` with a scheduler, ensuring the `data/` directory is updated with the latest end-of-day files.

## Roadmap
- Integrate additional macroeconomic features (rates, FX) as Mapper lenses to enhance regime discrimination.
- Extend meta-blend to incorporate PH regime features and forward-looking risk metrics in the learning set.
- Add unit tests for `scripts/` entrypoints and expand coverage for expected-return tilting edge cases.
- Containerise the environment for consistent cloud execution and CI automation.

---
For questions or support, reach out to the research maintainer or open an issue in the repository.
