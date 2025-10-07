# 09_Hull_Risk_notes.md
**Livro:** *Risk Management and Financial Institutions* — John C. Hull (4ª ed., Wiley, 2015)  
**Objetivo:** síntese executiva, fórmulas‑chave e aplicações práticas para uso no projeto **Quant Challenge**.  
**Escopo:** foco em tópicos que dialogam com AFML, APM, HRP, Liquidez/Execução e Regulação (Basel/FRTB).

---

## 1) Visão geral e mapa do livro
Hull organiza o tema em seis blocos:
1. **Instituições financeiras e negociação:** funções de bancos/seguradoras/fundos; estrutura de mercados; crise de 2007; distinção “mundo real” vs. “neutro ao risco”.
2. **Risco de mercado:** *Greeks*, risco de taxa, volatilidade (EWMA/GARCH), correlações & cópulas, VaR/ES (definição, teste, alocação), *historical simulation* & EVT, modelos paramétricos/MC.
3. **Regulação:** Basel I/II/II.5/III e **FRTB** (Expected Shortfall, P&L attribution, NMRF, *non‑modellable*).
4. **Risco de crédito:** gestão (margem/OTC/CCP), PD/recuperação (ratings, spreads, CDS, Merton), **CVA/DVA**, *Credit VaR* (Vasicek, CreditRisk+, CreditMetrics).
5. **Tópicos adicionais:** *stress testing*, risco operacional (LDA), liquidez (trading × funding), risco de modelo, capital econômico & **RAROC**, **ERM**, erros a evitar.
6. **Apêndices:** curvas zero, forwards, swaps, Black‑Scholes, séries de Taylor, PCA, matrizes de transição, valuation de CDS/CDO.

> **Uso no pipeline:** embasar controles de risco (VaR/ES, *stress*, liquidez), backtests, alocação por Euler, e requisitos regulatórios (Basel/FRTB) nos módulos `risk_controls.py`, `kpis.py`, `robustness.py` e `capacity.py`.

---

## 2) Conceitos‑âncora
- **Trade‑off risco‑retorno:** fronteira eficiente; CAPM/APT como baseline de retorno exigido; mas **instituições** gerem **risco total** (não só sistemático) por falência/regulação.
- **Medidas de risco:** **VaR** (percentil) vs. **ES** (média das perdas na cauda). ES é coerente; VaR não é subaditivo. FRTB troca VaR(99%) por ES(97.5%).  
- **Agregação & alocação:** *marginal/incremental/component risk*; **Teorema de Euler** para decompor ES/VaR por fator/ativo.
- **Backtesting:** exceções (Kupiec) para VaR; pontos de alavanca e *conditional coverage*; para ES, *ES backtesting* via *regression* e *elicitability* (conceito moderno).
- **Volatilidade:** retorno não normal, *power law*; **EWMA** (λ≈0.94 diários); **GARCH(1,1)** para previsão de σ².
- **Dependência:** correlação instável; **cópulas** (t‑cópula) para caudas; Vasicek (fator único gaussiano) para carteiras de crédito.
- **Liquidez:** *trading* vs. *funding*; buracos de liquidez, *fire sales*; incluir *haircuts*, *market impact* e *margin calls* em cenários de estresse.
- **Crédito & OTC:** colateralização (VM/IM), CSA, central clearing (**CCP**), **wrong‑way risk** e *gap risk*; *margin period of risk*.
- **Model risk:** *mark‑to‑model*, validação independente, *benchmarking*, *P&L explain*.  
- **Capital:** regulatório (Basel) vs. **econômico** (interno); **RAROC** para precificação e alocação.

---

## 3) Fórmulas e receitas rápidas

### 3.1 VaR & ES (linear, normal)
- **VaR\_{α}** = μ\_P + z\_{α} · σ\_P (para perdas: usar sinal e convenção consistentes)  
- **ES\_{α}** = μ\_P + σ\_P · ϕ(z\_{α}) / (1−α)  \[com ϕ densidade Normal\]  
- **Agregação (covariâncias):** σ\_P² = wᵀΣw  
- **Alocação por Euler (com ES):** contribuição\_i = w\_i · ∂ES/∂w\_i

### 3.2 EWMA & GARCH
- **EWMA:** σ²\_{t|t−1} = λ σ²\_{t−1|t−2} + (1−λ) r²\_{t−1}  (λ≈0.94 diário)  
- **GARCH(1,1):** σ²\_t = ω + α r²\_{t−1} + β σ²\_{t−1}  (α+β < 1; *half‑life* ≈ ln2/ln(1/(α+β)))

### 3.3 Vasicek – carteira de crédito (fator único)
- **PD condicional:** Φ( (Φ^{-1}(PD) − √ρ · Y) / √(1−ρ) )  
- **Perda de portfólio:** via integração MC/analítica; **Credit VaR** no quantil α da distribuição de perdas.

### 3.4 CVA simplificado (discreto)
- **CVA ≈ (1−R) · Σ\_t DF(t) · EE⁺(t) · PD(t\_{−},t)**  
  onde *EE⁺* é **expected positive exposure**, *R* recuperação, *PD* incremental.

### 3.5 Duração/Convexidade (IR risk)
- **ΔP/P ≈ −D\_mod · Δy + ½ C · (Δy)²**; usar PCA (level/slope/curvature) para choques não paralelos.

---

## 4) Mercado: gestão de risco do “trader”
- **Greeks:** Δ, Γ, Vega, Θ, Rho; *Taylor hedging*; custos de re‑hedge, saltos, *vol-of-vol* e *smile*.  
- **Exóticos:** *path‑dependence* complica hedge; *scenario analysis* como complemento.  
- **P&L attribution:** separar *risk‑theoretical P&L* vs. *residual*; base para FRTB e validação.

---

## 5) Volatilidade, correlação e caudas
- **Não‑normalidade:** assimetria e curtose elevam cauda → ES>VaR.  
- **Estimadores:** *rolling*, EWMA, GARCH; escolha via *likelihood* e poder preditivo.  
- **Cópulas:** usar t‑cópula para dependência em cauda; calibrar ν (graus de liberdade).  
- **EVT:** *Peaks over Threshold* (GPD) para cauda; importante em HS com janelas curtas.

---

## 6) VaR/ES: modelos e backtesting
- **Historical Simulation (HS):** simples, mas sofre com regime e volatilidade variável → *filtered HS*.  
- **Paramétrico (Delta‑Normal/Quadrático):** rápido; exige normalidade ou *Cornish‑Fisher*; sensível à matriz Σ.  
- **Monte Carlo:** flexível (sourcings não‑normais, sorrisos); custo computacional.  
- **Backtesting:** proporção de *breaches*; *traffic light*; *conditional coverage*; para ES, testes baseados em *joint elicitability*.  
- **Alocação:** **Euler** para ES (coerente) → *component ES* para limites e *RAROC*.

---

## 7) Regulação: Basel & FRTB (essencial)
- **Basel I/II/II.5/III:** evolução de capital de mercado, crédito (IRB), *stressed VaR*, *IRC* e *CVA capital*.  
- **FRTB (BCBS 2016+):**  
  - Substitui VaR(99%) por **ES(97.5%)** com **liquidity horizons** por bucket de risco.  
  - **P&L Attribution** e **Backtesting** para usar IMA; caso contrário, **SA** (standardized approach).  
  - **NMRF:** riscos não modeláveis → *add‑on* por *stress scenario risk measure*.  
  - *Boundary* Trading vs. Banking Book, e restrições de *desk*.

---

## 8) Crédito: PD, spreads, CDS, CVA/DVA
- **Estimando PD:** *through‑the‑cycle* (ratings históricos), *point‑in‑time* (spreads/EDFs), **Merton** via equity.  
- **Recuperação:** LTC vs. market‑implied; sensível a ciclo/setor/senioridade.  
- **CDS/Spreads:** bootstrap para curva de hazard; *basis* e liquidez.  
- **CVA:** risco de contraparte é material; *wrong‑way risk*; *hedge* parcial via CDS/indices; **CVA VaR**.  
- **DVA:** ajuste próprio (polêmico para risco de não performance);
- **Credit VaR:** **Vasicek**, **CreditRisk+**, **CreditMetrics** – diferentes hipóteses para dependência/ severidade.

---

## 9) *Stress testing*, risco operacional & liquidez
- **Cenários:** históricos (1987, 2008, taper tantrum, COVID‑like), hipotéticos e reversos (*reverse stress*).  
- **Operacional (AMA/LDA):** frequência×severidade, *power law*, *extreme tails*; controles preventivos.  
- **Liquidez:** 
  - **Trading liquidity:** *bid‑ask*, *market impact*, *black holes*.  
  - **Funding:** *run risk*, *wholesale funding*, *liquidity coverage* (LCR/NSFR).  
  - Integrar nos choques de preço e haircuts no *risk engine*.

---

## 10) Risco de modelo, capital econômico e RAROC
- **Ciclo de vida do modelo:** desenvolvimento → validação independente → *monitoring* contínuo.  
- **Erros comuns:** *overfitting*, mau mapeamento de riscos, parâmetros não robustos, *P&L explain* fraco.  
- **Capital econômico:** quantil (ex. 99.9% a 1 ano) por risco → agregação com correlações/pilares → **RAROC** = lucro ajustado ao risco / capital.  
- **ERM:** apetite a risco, cultura, limites top‑down, *aggregation* e *allocation* consistentes.

---

## 11) Checklists práticos (para o projeto)
- **Antes de rodar VaR/ES:** validar mapping, Σ/PCA atualizados, *vol filter*, limites por *component ES*.  
- **Crédito/CVA:** netting/CSA corretos, *EE profile*, wrong‑way, *stressed PD/LGD*.  
- **Liquidez:** horizonte por bucket (FRTB), *haircuts* e *market impact* nos cenários.  
- **Backtesting:** janelas e *exceptions*; *traffic light*; *alerts* automáticos → `kpis.py`.  
- **Governança:** *model inventory*, versões, *P&L attribution*, documentação.

---

## 12) Tabela‑relâmpago (parâmetros de referência)
| Tema | Regra/Valor típico |
|---|---|
| EWMA λ (diário) | 0.94 |
| ES vs VaR (Normal) | ES ≈ 1.25×VaR (α=97.5% vs 99% muda) |
| FRTB ES | 97.5% com *liquidity horizons* |
| PCA taxas | 3 fatores (level/slope/curvature) explicam >95% |
| Horizonte VaR *trading* | 1–10 dias (FRTB usa horizontes por risco) |
| Backtesting VaR | Exceções compatíveis com α (ex.: 1% → ~2–3/mês ~ 250d) |

---

## 13) Pontos de ligação com outros *notes*
- **04_APM_notes:** CAPM/APT, alfa/beta e avaliação de gestores.  
- **01_AFML_notes / 02_ML4AM_notes:** previsão de retorno/vol, validação *out‑of‑sample*, *purged CV*.  
- **12_Unification_HRP_MinVar_notes:** fronteira eficiente, *risk parity*, ligação com ES/Euler.  
- **08_AlmgrenChriss_notes:** *market impact* e execução → risco de liquidez.

---

## 14) Perguntas de checagem (auto‑teste)
1. Por que **ES** é preferível a **VaR** em coerência e agregação?  
2. Diferencie **trading liquidity** e **funding liquidity** e como testá‑las em *stress*.  
3. Como alocar **ES** por fator/desk via **Euler**?  
4. Quais pressupostos distinguem **HS**, **Delta‑Normal** e **MC**?  
5. Dê um exemplo de **wrong‑way risk** em CVA e como mitigá‑lo.  
6. O que muda do **Basel II.5** para **FRTB** na mensuração de risco de mercado?  
7. Compare EWMA e GARCH(1,1). Quando preferir cada um?  
8. Explique a lógica do modelo de **Vasicek** e sua limitação.  
9. Quais pilares de **ERM** e como conectam a limites operacionais?  
10. Cite três erros clássicos de **model risk** e defesas correspondentes.

---

**Resumo executivo:** Hull é o *playbook* de risco para instituições: mede (VaR/ES), testa (backtests/stress), precifica (CVA/RAROC), regula (Basel/FRTB) e governa (ERM/Model Risk). No **Quant Challenge**, use estes blocos para: (i) desenhar KPIs e limites; (ii) integrar liquidez/execução; (iii) alinhar modelos a requisitos de capital e auditoria.

