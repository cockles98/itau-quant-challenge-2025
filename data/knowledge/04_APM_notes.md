# 04_APM_notes.md
**Reference**: Richard C. Grinold & Ronald N. Kahn — *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed., 1999).

## TL;DR
The book formalizes how skill (IC), number of independent bets (breadth), and implementation (transfer coefficient) combine to determine achievable information ratio (IR). It ties forecasting, risk modeling, and optimization into one loop and shows how constraints, costs, and turnover reduce theoretical edge.

---

## 1) Core Framework (What you’re optimizing)
- **Active return**: excess vs benchmark.  
- **Tracking error (TE)**: std. dev. of active return.  
- **Information Ratio (IR)**: \( IR = \frac{\mathbb{E}[R_{active}]}{TE} \).  
- **Optimizer ideal (unconstrained)**: with alpha vector \(\alpha\) and covariance \(\Sigma\), the optimal active weights are proportional to  
  \[ w^{\*} \propto \Sigma^{-1}\,\alpha. \]
- In practice, constraints (long-only, bounds, sectors) and costs mean you solve a **constrained mean–variance** problem for active risk, not the frictionless formula.

### Fundamental Law of Active Management
- **Basic form**: \( IR \approx IC\,\sqrt{BR} \)  
  where **IC** is the information coefficient (forecast–outcome correlation) and **BR** is the **breadth** (number of *independent* bets).
- **With constraints/implementation**: \( IR \approx TC\cdot IC\cdot \sqrt{BR} \), where **TC** is the **transfer coefficient** (0–1) capturing how much of the signal gets into the portfolio after constraints/costs.
- **Effective breadth**: correlations between bets/overlapping horizons **reduce BR**; use **effective N** rather than raw count.

---

## 2) Forecasts & IC (Where edge comes from)
- **Alpha sources**: valuation, momentum, quality, themes, analyst views, etc. Combine with **cross‑sectional standardization** and **winsorization**.
- **IC**: correlation between forecast and realized active return over a horizon.  
  - Expect **IC decay** as the holding period lengthens; horizon alignment is key.  
  - Combine multiple signals with (Bayesian or regression) **IC‑weighted** blending; penalize multicollinearity.
- **Breadth**: more *independent* bets (assets, time, strategies) increase IR via \(\sqrt{BR}\); overlapping signals/timeframes inflate BR on paper but not in reality.
- **Transfer coefficient (TC)** goes down with tight bounds, risk budgets, turnover caps, and capacity limits.

---

## 3) Risk Model (What can go wrong and how big)
- **Factor model**: \( r = B f + \epsilon \), with factor covariance and **specific risk**.  
- **Covariance**: stabilize with **shrinkage**, clamping, and **out‑of‑sample (OOS)** validation.  
- Use **tracking‑error budgets** to size active risk and **ex‑ante** IR forecasts.
- **Constraint risk**: long‑only, sector/position caps, and benchmark exposures change risk contribution and TC.

---

## 4) Portfolio Construction (Turning scores into weights)
- Solve constrained optimization for active return vs TE, incorporating **transaction costs** (linear + nonlinear impact).  
- **Rebalancing**: trade only when expected benefit > cost (use **implementation shortfall**).  
- **Turnover control**: explicit caps and signal smoothing; recognize the **IC–turnover** trade‑off.  
- **Capacity**: limit participation vs ADV; expect IR to drop as you push size (TC↓, costs↑).

---

## 5) Performance Measurement & Attribution
- **IR vs Sharpe**: IR uses **relative** risk (TE); Sharpe uses total volatility.  
- **t‑stat of IR**: time needed to verify skill scales with \(1/IR^2\).  
- **Attribution**: decompose active return into factor, stock‑specific, timing, constraints, and costs.

---

## 6) How this maps to *Atlas* (T‑HRP v3.0) in this project
- **Forecasts (\(\alpha\))**: our blend = **Regime/TDA** + **Momentum** + **Quality** (α,β,γ). Cross‑sectional z‑scores & winsorization → APM‑consistent preprocessing.
- **Risk model**: we use rolling covariances + **HRP** ordering/alloc to produce stable weights **without inverting** \(\Sigma\) directly—an APM‑aligned goal (stability OOS) via a different mechanism.
- **From scores to weights**: ideal APM says \(w\propto \Sigma^{-1}\alpha\); we approximate with **HRP weights** adjusted by the signal mix, then **ATR normalization** and **vol targeting** → disciplined risk budget.
- **TC in our pipeline**: **cluster/asset caps**, **turnover caps**, **participation caps**, and **cost model** are explicit levers; we monitor **bind rate** and **cut fraction** as proxies for TC loss.
- **Breadth**: cross‑sectional (many assets) + temporal (walk‑forward retraining). We avoid over‑counting by checking **effective** BR via correlation/overlap diagnostics.
- **Capacity & costs**: APM’s guidance → our capacity curves (Sharpe×TE vs ADV participation) and stress tests.

---

## 7) Practitioner Checklist
- [ ] Align **forecast horizon** with rebalance and TE measurement window.  
- [ ] **Standardize & winsorize** alphas; combine by IC/variance.  
- [ ] Use a **stable covariance** (shrinkage/HRP order) and validate OOS.  
- [ ] Encode **constraints** explicitly; track their impact on TC.  
- [ ] Model **costs & impact**; trade only when edge > cost.  
- [ ] Monitor **IR, TE, turnover, capacity**, and attribution.  
- [ ] Re‑estimate **IC, decay, and BR** periodically; beware data‑mining.

---

## 8) Pitfalls & Anti‑patterns
- Overstated BR (non‑independence), ignoring IC decay, using noisy \(\Sigma\), tight bounds that zero out TC, neglecting costs, and confusing Sharpe with IR when benchmarked mandates dominate.

---

## 9) Key Equations (quick ref)
- \( IR \approx TC\cdot IC\cdot \sqrt{BR} \)  
- \( w^{\*} \propto \Sigma^{-1}\alpha \) (unconstrained ideal)  
- **Ex‑ante IR** from optimizer: use forecasted alpha and TE from risk model; compare to realized IR OOS.

---

## 10) Further Reading
- Grinold & Kahn (2e), ch. 6–9 (Fundamental Law, Forecasting, Breadth), ch. 10–13 (Risk Models & Optimization), ch. 17 (Performance).  
- Clarke, de Silva & Thorley — “Portfolio Constraints and the Fundamental Law.”  
- Ait‑Sahalia & Lo — **Implementation Shortfall** and trading costs.

> **Usage note**: This file follows the same structure as the other `/knowledge` notes. Drop the next title/PDF and we’ll mirror this template.
