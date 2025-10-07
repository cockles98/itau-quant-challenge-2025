# Optimal Execution (Almgren–Chriss, 2000) — Notas para o Atlas

tags: [execution, market-impact, temporary, permanent, efficient-frontier, half-life, L-VaR, serial-correlation, regime-shift, multi-asset]

## 1) Tese (80/20)
Define um **modelo de execução ótima** que troca **custo esperado** por **risco (variância) do custo**. Com impacto **permanente** e **temporário** lineares na taxa de negociação, o problema vira **quadrático** e produz:
- **Fronteira eficiente** de estratégias (mínimo custo esperado dado um nível de variância).  
- **Trajetória ótima estática** (pré-planejada) sob retornos i.i.d. e utilidade quadrática.  
- **Meia‑vida (θ)** da negociação: tempo intrínseco de liquidação determinado por vol, fricções e aversão a risco.  
- Extensões: **VaR de liquidação (L‑VaR)**, **drift** e **autocorrelação** de curto prazo, **eventos programados** (regime shifts) e **carteira multi‑ativo**.

---

## 2) Modelo (discreto, N passos)
- **Dinâmica do preço** (sem drift básico):  
  \(S_k = S_{k-1} + \, \sigma \, \sqrt{\tau}\,\xi_k \, - \, \tau\, g(v_k)\), com \(v_k = n_k/\tau\).
- **Impacto permanente**: \(g(v) = \gamma v\).  
- **Impacto temporário**: \(h(v) = \varepsilon\,\mathrm{sgn}(v) + \eta v\).  
- **Preço efetivo** no k‑ésimo lote: \(\tilde S_k = S_{k-1} - h(v_k)\).
- **Custo esperado** (shortfall) e **variância** da trajetória \(x_0\to x_N\):  
  \[
  E[x] = \tfrac12\gamma X^2 + \varepsilon\sum|n_k| + \frac{\tilde\eta}{\tau}\sum n_k^2, \quad
  V[x] = \sigma^2 \sum \tau x_k^2, \quad \tilde\eta=\eta-\tfrac{\gamma\tau}{2}.
  \]

---

## 3) Fronteira eficiente e trajetória ótima
- Resolver \(\min_x \big(E[x]+\lambda V[x]\big)\) ⇒ EDO discreta linear \(x_{k-1}-2x_k+x_{k+1}=\tilde\kappa^2\tau^2 x_k\).  
- **Solução** (venda monótona):  
  \(x_j = \dfrac{\sinh\big(\kappa (T-t_j)\big)}{\sinh(\kappa T)}\,X\),  
  \(n_j \propto \cosh\big(\kappa(T-t_j-\tfrac{\tau}{2})\big)\).
- **Escala do problema**:  
  \(\kappa \approx \sqrt{\lambda\,\sigma^2/\eta}\), **meia‑vida** \(\theta=1/\kappa\).  
  - Se **\(T\ll\theta\)** ⇒ reta quase linear (minimiza custo).  
  - Se **\(T\gg\theta\)** ⇒ vende cedo (minimiza risco).

> **Intuição**: \(\lambda\) (aversão a risco) e \(\sigma\) puxam para vender rápido; \(\eta\) (impacto temp.) puxa para suavizar.

---

## 4) Seleção via Utilidade × L‑VaR
- **Utilidade quadrática**: minimizar \(E+\lambda V\) dá o ponto ótimo na fronteira.  
- **Value‑at‑Risk de liquidação (L‑VaR)**: \(\text{VaR}_p(x)=E[x]+\lambda_v\sqrt{V[x]}\).  
  - Para um nível \(p\), o ótimo é a **reta tangente** à fronteira no plano \((\sqrt V,E)\).

---

## 5) Informação extra: drift, autocorrelação, eventos
- **Drift** (\(\alpha\neq 0\)): solução soma **trajetória sem drift** + **correção** que retarda/veste posição até nível \(\bar x=\alpha/(2\lambda\sigma^2)\).  
- **Autocorrelação 1‑passo** (\(\rho\)): mover \(\delta n\) entre períodos rende ganho \(\approx \rho^2\sigma^2\tau^2/(4\eta)\) por período; **independente do tamanho do bloco** (linearidade do impacto).  
- **Evento programado** (earnings, \(T_*\)): estratégia **piecewise estática**: planeja prefixo até \(T_*\), observa regime \(R_j\), e segue a trajetória ótima condicional ao novo \(\{\sigma,\eta,\ldots\}\).

---

## 6) Carteira multi‑ativo (resumo)
- Vetores/matrizes: **vol** \(C=\sigma\sigma^T\), **impacto perm.** \(\Gamma\), **temp.** \(H\).  
- Com \(\Gamma, H\) diagonais (impacto idiossincrático), solução desacopla nas EDOs via autodecomposição de \(\lambda\tilde H^{-1}C\); correlação em \(C\) acopla o **timing** entre ativos (p.ex. vender o líquido mais cedo para reduzir risco).

---

## 7) Dials práticos no Atlas
- **Parâmetros de impacto** (\(\varepsilon,\eta,\gamma\)): calibrar por **%ADV vs. spread**; stressar 0.5×/1×/2×.  
- **Aversão a risco \(\lambda\)**: ligar ao **regime TFI** (↑TFI ⇒ ↑\(\lambda\) ⇒ ↓θ).  
- **Horizonte/grade**: granularidade \(\tau\) compatível com a janela de liquidez do universo; contínuo é limite teórico.  
- **Restrições**: caps de participação, curva de **meia‑vida** por ativo, limites de **turnover**.  
- **Telemetria**: logar \(\theta\), \(\kappa\), **L‑VaR**, custo ex‑ante/ex‑post, e deltas por evento.

---

## 8) Integração no código Atlas
- `costs.py`: \(E[x]\) e \(V[x]\) ex‑ante; funções para **L‑VaR** e stress.  
- `execution.py`: solver fechado para \(x_j\) dado \(\lambda,\sigma,\eta\) (+ caps) e rotina **piecewise** para eventos.  
- `engine.py`: mapear **regime TFI → λ/gross/target‑vol**; honrar **participation cap** e **kill‑switch**.  
- `kpis.py`: reportar **fronteira** (grid de \(\lambda\)), **θ** médio ponderado, **shortfall** e **tracking** real vs. alvo.

---

## 9) Checklist de validação
- [ ] Fronteira convexa bem‑comportada; ponto \(\lambda=0\) ≈ reta (min‑custo).  
- [ ] **θ** cai quando (\(\lambda,\sigma\)) sobem e sobe quando \(\eta\) sobe.  
- [ ] **L‑VaR** consistente com alvo de confiança (p.ex. 95%).  
- [ ] Ganho por **autocorrelação** é pequeno vs. custo de impacto (sanity check).  
- [ ] Em eventos, trajetória **piecewise** bate a estratégia ingênua.  
- [ ] Multi‑ativo: timing reduz risco agregado conforme \(C\).

---

## 10) Referências
- Almgren & Chriss (2000), **Optimal Execution of Portfolio Transactions**.  
- Bertsimas & Lo (1998) — execução dinâmica (contraste com fronteira estática).  
- Perold (1988) — **Implementation Shortfall** (métrica).

