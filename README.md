# 🗺️ Atlas: A Regime-Aware Quantitative Strategy
<div align="center">

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)
![TDA](https://img.shields.io/badge/Method-Topological_Data_Analysis-purple?style=for-the-badge)

</div>

> **Hierarchical Risk Parity with regime detection using Topological Data Analysis (TDA)**
> *Submission for the Itaú Quant Challenge 2025 — Top 4% (40/953)*

**Atlas** is a long-only quantitative allocation framework that uses *Topological Data Analysis* and *Machine Learning* to navigate different market regimes. Unlike traditional models based solely on linear correlations — which tend to break down exactly when you need them most, during crises — Atlas uses *Persistent Homology* to detect structural market turbulence and *TDA/Mapper* to cluster assets by topological behavior. From there, a *Meta-Blend (ML ensemble)* dynamically determines the optimal factor weights (Momentum, Quality, Carry) for the current regime, and *Hierarchical Risk Parity (HRP)* builds the final portfolio.

---

## 🚀 Performance & Results
> **Period:** 09/14/2017 to 10/06/2025 (Ibovespa Universe)
> **Validation:** Walk-Forward Analysis (504d Train / 126d Test)

Atlas consistently outperformed both the benchmark (Ibovespa) and the CDI risk-free rate, delivering strong risk-adjusted returns with downside protection via an automated kill-switch.

Equity Curves (Walk-Forward vs. Ibovespa):

![Equity Curve](data/readme_assets/equity_curve_comparison_v2.png)

Key Performance Indicators (KPIs):

| Metric | Atlas (Walk-Forward) | Ibovespa |
| :--- | :--- | :--- |
| **CAGR** | **0.316** | 0.209 |
| **Sharpe** | **1.181** | 0.549 |
| **Sortino** | **1.903** | 0.813 |
| **Vol** | **0.166** | 0.234 |
| **MaxDD** | **-0.199** | -0.254 |
| **AvgTimeUnderWater** | **20.400** | 30.800 |
| **MaxTimeUnderWater** | **270.000** | 500.000 |
| **Calmar** | **1.590** | 0.821 |
| **HitRate** | **0.542** | 0.515 |
| **Turnover** | **0.017** | NaN |

KPIs per out-of-sample (OOS) window from the walk-forward analysis:

| Metric | WF 01 | WF 02 | WF 03 | WF 04 | WF 05 | WF 06 | WF 07 | WF 08 | WF 09 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAGR** | 1.284 | 0.059 | -0.031 | 0.553 | 0.221 | 0.413 | 0.187 | 0.187 | 0.312 |
| **Sharpe** | 5.080 | 0.254 | -0.360 | 1.698 | 0.706 | 1.302 | 0.522 | 0.547 | 0.882 |
| **Sortino** | 11.760 | 0.401 | -0.448 | 5.643 | 1.307 | 2.051 | 0.836 | 0.818 | 1.281 |
| **Vol** | 0.161 | 0.176 | 0.167 | 0.197 | 0.110 | 0.184 | 0.147 | 0.158 | 0.173 |
| **MaxDD** | -0.039 | -0.099 | -0.132 | -0.033 | -0.053 | -0.097 | -0.058 | -0.069 | -0.094 |
| **AvgTimeUnderWater** | 6.820 | 19.000 | 0.000 | 14.750 | 9.200 | 21.400 | 12.560 | 6.860 | 11.670 |
| **MaxTimeUnderWater** | 29.000 | 37.000 | 0.000 | 45.000 | 23.000 | 57.000 | 36.000 | 19.000 | 60.000 |
| **Calmar** | 32.776 | 0.591 | -0.234 | 16.515 | 4.126 | 4.275 | 3.222 | 2.693 | 3.310 |
| **HitRate** | 0.691 | 0.540 | 0.489 | 0.519 | 0.516 | 0.553 | 0.517 | 0.532 | 0.526 |
| **Turnover** | 0.022 | 0.021 | 0.029 | 0.016 | 0.008 | 0.011 | 0.017 | 0.007 | 0.017 |

Walk-forward outperformance vs. Ibovespa (cumulative excess return):

![Cumulative Excess Return](data/readme_assets/benchmark_curve.png)

Relative KPIs (Atlas vs. Ibovespa):

| Metric | Atlas vs Ibovespa |
| :--- | :--- |
| **Annualized Alpha** | 0.158 |
| **Beta** | 0.269 |
| **Excess Annual Return** | 0.064 |
| **Tracking Error** | 0.231 |
| **Information Ratio** | 0.277 |
| **Correlation** | 0.377 |

---

## 🧠 The Innovation: Why Topology?

Traditional quant models rely on linear correlations — but correlations converge to 1 during crises, precisely when diversification is needed most. Atlas addresses this with three core engines:

### 1. Turbulence Detector (Persistent Homology)
Instead of using simple volatility, Atlas measures the *shape* of the market's data cloud.
- **How it works:** *Vietoris-Rips filtration* tracks the persistence of topological "holes" in the market's return space.
- **Practical effect:** When the topological structure breaks — a signal of systemic stress — the algorithm automatically activates *Risk-Off* mode, reducing exposure before volatility spikes.

### 2. Asset Clustering via Mapper
Assets are grouped not just by sector or correlation, but by topological behavior.
- **The differentiator:** The *Mapper* algorithm projects assets onto a graph, identifying which stocks are "peripheral" (idiosyncratic/safe) and which are "central" (systemic/risky).
- **Application:** The portfolio tilts toward peripheral assets during periods of uncertainty.

### 3. Meta-Blend (Machine Learning)
An ensemble model (Ridge/ElasticNet) that dynamically learns the optimal mix of factors (Momentum, Quality, Carry) for the current regime detected by the topology.

---

## 🛠️ Engineering & Reproducibility

The project was built with software engineering rigor to be fully auditable and reproducible.

### Execution Pipeline
1. **Ingestion:** Data loading and universe filtering (Liquidity/Hysteresis).
2. **TDA Engine:** Computation of *Persistence Landscapes* and *Mapper* graphs.
3. **ML Layer:** *Meta-Blend* training with *Purged K-Fold Cross Validation*.
4. **Portfolio Optimization:** HRP guided by the topological structure.
5. **Risk Guards:** Drawdown-based kill-switch and turnover control.

### Getting Started
Environment managed via `uv` or `pip`. Requires Python 3.10+.

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# 2. Run Full Backtest
python -m src.main --mode backtest --config configs/base.yaml

# 3. Generate PDF Report
python -m src.reports.build_pdf --config configs/base.yaml
```

### Python Usage
Run the full pipeline from a script or notebook without the CLI:

```python
from pathlib import Path

from src.backtest.engine import run_backtest
from dataio.config import load_config
from dataio.loaders import get_panel

cfg = load_config("configs/base.yaml")
panel = get_panel(cfg["dates"]["start"], cfg["dates"]["end"])

result = run_backtest(cfg, panel=panel)
print(result["kpis"])

equity_path = Path("reports") / "equity_curve_example.csv"
result["equity_curve"].to_csv(equity_path)
print(f"Equity curve saved to: {equity_path}")
```

You can also run everything via `notebooks/full_pipeline.ipynb`, which reproduces the official pipeline used to generate the reported results.

## TDA/Mapper Animation
Visualization of the topological evolution (Mapper) over time:

![Mapper TDA](data/readme_assets/mapper_tda.gif)

---

## 📂 Repository Structure

```text
.
├── configs/           # YAML files (model hyperparameters)
├── src/
│   ├── features/      # TDA (Mapper/PH) and feature engineering
│   ├── models/        # Meta-models (ElasticNet/Ridge)
│   ├── backtest/      # Execution engine and HRP
│   └── risk/          # Risk management and kill-switches
├── artifacts/         # Generated outputs (caches, saved models)
└── reports/           # PDFs, metric CSVs, and final charts
```

---

## 🔎 Robustness & Validation

To ensure results are not due to overfitting or p-hacking:

- **Deflated Sharpe P-Value:** 11.6% — meaning 88.4% confidence that Sharpe > 0 is not noise.
- **Sensitivity Analysis:** Model maintains stable performance across a wide range of `target_vol` (12–14%).
- **Realistic Costs:** Simulation includes non-linear slippage and brokerage fees.

---

## 📄 License

© 2025 Atlas Project.

Licensed under the MIT License. You are free to use, modify, and distribute this software with attribution. See [LICENSE](LICENSE) for the full text.

---

*Interested in similar work — quantitative modeling, backtesting, or risk analysis? Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/felipe-cockles) or [email](mailto:felipe.cockles@hotmail.com).*
