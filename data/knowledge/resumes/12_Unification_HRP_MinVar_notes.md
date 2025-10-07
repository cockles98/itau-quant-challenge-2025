# Unification HRP ↔ MinVar (HMV) — Notas para o Atlas

tags: [hrp, hmv, schur, minimum-variance, covariance, seriation, recursion, gamma]

## 1) Tese (80/20)
O paper propõe **HMV — Hierarchical Minimum Variance**, uma família de alocações hierárquicas
que **interpolam** entre **HRP** (γ=0) e **Mínima Variância** (γ=1), usando **complementos de Schur**
para levar **informação off-block** (A–B–D) para dentro de cada subcarteira. Na prática,
o método mantém o “divide-and-conquer” do HRP, mas **reinsere parte das covariâncias** que
o HRP ignora, reduzindo viés de quebra de bloco e recuperando a **simetria** em casos patológicos.

---

## 2) Ideia central (receita)
Dada a matriz \(\Sigma\) reordenada por uma **seriação** (árvore/TDA), partimos da bisseção
\(\Sigma = \begin{pmatrix}A & B\\ C & D\end{pmatrix}\). Em HMV, ao descer a recursão, **substituímos**
as submatrizes por versões **aumentadas** que incorporam dependências cruzadas:

- **Inter-grupos** (para decidir o split de capital): use \(A'\) e \(D'\) definidos por
  \[ A' = (A_c)^{\*b_A}, \quad A_c = A - \gamma B D^{-1} C, \quad b_A = \mathbf{1} - \gamma B D^{-1} \mathbf{1} \]
  (análogos para \(D'\)). O operador \((\cdot)^{\*b}\) denota \(Q^{\*b} = \big(Q^{-1}\cdot (bb^T)\big)^{-1}\).

- **Intra-grupos** (para alocar *dentro* de A e D): use \(A''\) e \(D''\) com
  \[ A'' = \dfrac{A - \gamma B D^{-1} C}{b_A b_A^T} \quad \text{(divisão elemento a elemento)}. \]

- **Recursão**: em cada nó, aloque entre os blocos pela “aptidão inversa” \(\nu(\cdot)\) (tipicamente a **variância de portfólio min-var** do bloco), e desça usando \(A''\) e \(D''\) até o critério de parada (por ex., tamanho \(\le m\)).

- **Contínuo γ**: \(\gamma\in[0,1]\) controla o quanto da informação cruzada (\(B\)) entra — \(\gamma=0\Rightarrow\) HRP clássico; \(\gamma=1\Rightarrow\) recupera MinVar (quando bem condicionado).

---

## 3) Equações-chave (para implementação)
- **Split recursivo equivalente à MinVar (esqueleto)**:  
  \[
    w(\Sigma) \propto \begin{pmatrix}
      \nu\big((A_c)^{\*b_A}\big)^{-1}\; w\!\left( \dfrac{A_c}{b_A b_A^T} \right) \\[4pt]
      \nu\big((D_c)^{\*b_D}\big)^{-1}\; w\!\left( \dfrac{D_c}{b_D b_D^T} \right)
    \end{pmatrix}
  \]
  com \(A_c = A-\gamma BD^{-1}C\), \(b_A=\mathbf{1}-\gamma BD^{-1}\mathbf{1}\), análogos para \(D\).
- **Portf. de mínima variância com restrição \(w^Tb=1\)** (para definir \(\nu\)):  
  \(w(Q,b)=Q^{-1}b\,/\,(b^TQ^{-1}b)\) e \(\nu(Q,b)=1/(b^TQ^{-1}b)\).

---

## 4) Pseudocódigo (um nível da recursão)
```text
def hmv_block(Sigma, order, gamma, nu, wterm, stop_m):
    # 1) Reordenar Σ pela seriação (order) e bissecar: A,B,C,D
    A,B,C,D = split(Sigma, order)

    # 2) Construir complementos e vetores b (com γ)
    Ac = A - gamma * B @ inv(D) @ C
    Dc = D - gamma * C @ inv(A) @ B
    bA = ones(len(A)) - gamma * B @ inv(D) @ ones(len(D))
    bD = ones(len(D)) - gamma * C @ inv(A) @ ones(len(A))

    # 3) Inter-grupos: medir fitness inversa
    A_prime   = inv(inv(Ac) * (bA@bA.T))^-1      # (Ac)^{*bA}
    D_prime   = inv(inv(Dc) * (bD@bD.T))^-1
    aL = 1 / nu(A_prime) ; aR = 1 / nu(D_prime)
    split_capital = normalize([aL, aR])

    # 4) Intra-grupos (subalocação)
    A_pprime  = elementwise_div(Ac, (bA@bA.T))
    D_pprime  = elementwise_div(Dc, (bD@bD.T))

    wL = wterm(A_pprime) if dim(A)<=stop_m else hmv_block(A_pprime, order_L, gamma, nu, wterm, stop_m)
    wR = wterm(D_pprime) if dim(D)<=stop_m else hmv_block(D_pprime, order_R, gamma, nu, wterm, stop_m)

    return concat_and_scale(wL, wR, split_capital)
```

> **Notas práticas**: (i) verificar **PD** antes de inverter; se necessário, reduzir \(\gamma\) adaptativamente (backoff) e/ou aplicar **shrinkage** fraco; (ii) manter a mesma **seriação (TDA/HRP)**; (iii) escolher \(\nu\) = variância do min-var com *weak shrink*.

---

## 5) Dials de projeto (para o Atlas)
- **γ (ponte HRP↔MinVar)**: comece em 0.2–0.4; habilite *backoff* se \(A_c\) ou \(D_c\) não forem PD.
- **Shrinkage “fraco”**: reduzir levemente off-diagonais (ξ≈0.95–0.99) só para estabilizar MinVar/\(\nu\).
- **Seriação**: use a ordem vinda do **grafo TDA** (Mapper) que já usamos no T‑HRP: maximiza “quase‑diagonalização” e reduz ruído de B.
- **Parada/terminal (m)**: usar MinVar (com shrink) quando tamanho do bloco ≤ 5–10.
- **Rebalance**: igual ao HRP atual (ex.: mensal), com **caps** e **turnover cap**; monitorar se γ aumenta turnover.
- **Métricas**: comparar **variança OOS**, **ERC por cluster**, **MDD**, **turnover**, **PSR**.

---

## 6) Pitfalls & como mitigar
- **Violação de PD** em \(A_c,D_c\): adote *γ adaptativo* e/ou Ledoit–Wolf leve; garanta clip de autovalores.
- **Quebra de simetria/viés de corte**: HMV corrige boa parte (ex. 3 ativos simétricos) — teste unitário.
- **Custo computacional**: mais inversões; use fatoração de bloco, cache e limites de dimensão para wterm.
- **Overfitting em γ**: valide em **walk‑forward** com **purged CV + embargo** (já temos utilitários).

---

## 7) Integração com o Atlas (mapa do código)
- `hrp.py`: extrair a recursão atual e generalizar para **HMV** com parâmetro `gamma` e `nu` (callable).
- `engine.py`: expor `gamma` no YAML; telemetria de `gamma_eff` após backoff por PD; logar `nu(A'), nu(D')`.
- `robustness.py`: grid de `gamma ∈ {0, 0.2, 0.4, 0.6, 0.8}` × shrinkage ξ; heatmap de variança OOS/turnover.
- `kpis.py`: acrescentar **ERC por cluster** e **var. condicional pós-split** para diagnósticos.
- `position_sizing.py`: inalterado; o sizing ocorre após pesos HMV.

---

## 8) Checklist de validação
- [ ] HRP (γ=0) vs HMV (γ>0): **var OOS menor** e **MDD** não piora.
- [ ] **Turnover** similar ao HRP (dif. ≤ 10–20%). Se ↑, atuar com caps/temperatura softmax.
- [ ] **Simetria** nos casos de teste (3 ativos com ρ igual) → pesos iguais.
- [ ] **PD** garantida no caminho (log de *backoffs* de γ).
- [ ] **Stress de custos** e **capacidade** preservam ranking.

---

## 9) Referências rápidas
- Cotton (2024) — _Schur Complementary Allocation: A Unification of HRP and Minimum Variance_. Conceitos de **HMV**, recursão via **Schur**, contínuo **γ**, shrinkage fraco e exemplos.
- López de Prado (2016/2019) — HRP e discussão de estimação de correlações.
- Ledoit–Wolf (2003) — shrinkage de covariância.

