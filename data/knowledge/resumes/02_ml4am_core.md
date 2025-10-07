# ML4AM — Resumo Operacional para o Atlas (T‑HRP)
**Fonte**: *Machine Learning for Asset Managers* (Marcos M. López de Prado, Cambridge, 2020).  
**Objetivo**: destilar o 20% do livro que entrega 80% do valor prático para o projeto **Atlas** (T‑HRP com TDA + gestão de risco ativa).  
**Última atualização**: 2025-10-07

---

## 0) Como usar este arquivo
- **É um “manual de integração”**: conecta capítulos do ML4AM aos módulos do Atlas (arquivos `.py`) com *hooks* e *checklists*.
- **Pronto para “dietar contexto”**: se o GPT tiver janela curta, priorize as seções marcadas com 🔹 *Prioridade*.

---

## 1) Mapa 80/20 (o que realmente importar para o Atlas)
🔹 **Cap. 2 — Denoising & Detoning** → pré-processamento de matrizes de correlação/covariância antes do HRP/T‑HRP.  
🔹 **Cap. 3 — Distance Metrics (MI/VI)** → distâncias mais robustas que correlação para clustering/topologia.  
🔹 **Cap. 4 — Optimal Number of Clusters (ONC)** → escolher *K* automaticamente para árvores/partições.  
🔹 **Cap. 5 — Financial Labels** → *triple‑barrier* e *trend‑scanning* para rotular regimes/metarótulos.  
🔹 **Cap. 6 — Feature Importance (MDA/MDI vs p‑values)** → auditoria de sinais/alphas.  
🔹 **Cap. 7 — Portfolio Construction** → críticas a média‑variância; sinergia com HRP/T‑HRP.  
🔹 **Cap. 8 — Test‑Set Overfitting** → CPCV/deflated Sharpe para blindar validação.

> **Regra prática**: Denoise/Detone → Medir distância (idealmente MI/VI) → Clustering/ONC → Construção (HRP/T‑HRP) → Gestão de risco → Validação CPCV.


---

## 2) Ponte ML4AM → Atlas (arquivos e *hooks*)

### 2.1 Denoising & Detoning (Cap. 2)
**Problema**: matrizes empíricas estão contaminadas por ruído; shrinkage genérico pode matar sinal fraco.  
**Ação**:
- `hrp.py` → adicionar *flag* `use_mp_denoise=True`, `detone=True`.
- Implementar **Marčenko–Pastur fit** para estimar `lambda_plus` e separar autovalores de ruído vs sinal.
- **Denoise**: reconstruir correlação usando apenas autovetores de sinal; opcionalmente **detonar** removendo 1º PC (fator de mercado).
- **Teste A/B**: HRP/T‑HRP com/sem denoise; reportar cond. number, estabilidade de pesos e turnover.

**Pseudo‑pipeline**:
```python
# hrp.py (esqueleto)
def preprocess_corr(corr, use_mp_denoise=True, detone=True):
    if use_mp_denoise:
        evals, evecs = np.linalg.eigh(corr)
        lam_plus, sigma2 = fit_marcenko_pastur(evals)  # estimador robusto
        mask = evals > lam_plus
        corr = rebuild_from_eigens(evals[mask], evecs[:, mask])
    if detone:
        corr = remove_first_pc(corr)
    return force_corr(corr)  # projetar em espaço de correlações válidas
```

**Checklist**:
- Estime **q = T/N** e use janela rolante consistente com o *use-case*.
- Salve `n_signal_factors` por *timestamp* para diagnóstico.
- Reporte impacto nos **pesos HRP** e na **alocação de risco por cluster**.

---

### 2.2 Distance Metrics (Cap. 3)
**Objetivo**: substituir a distância de correlação por **informação mútua (MI)** / **variation of information (VI)** quando possível.  
**Ação**:
- `factors.py` ou novo `metrics.py`: implementar MI/VI (com discretização robusta ou estimador k‑NN).  
- `tda.py` / `hrp.py`: aceitar `distance="vi" | "1-corr" | "custom"`.

**Hook**:
```python
def pairwise_distance(X, metric="vi"):
    if metric == "vi":
        return vi_matrix(X)  # simetrizar, normalizar
    elif metric == "mi":
        return mi_to_distance_matrix(X)
    else:
        return corr_to_distance(X)  # sqrt(0.5*(1 - rho))
```

**Checklist**:
- Quantilizar/discretizar séries **sem vazar futuro**.
- Auditar estabilidade de *ranking* de distâncias sob bootstrap.

---

### 2.3 Optimal Clustering (Cap. 4, ONC)
**Objetivo**: recuperar **número e composição** de clusters sem *guess* de K.  
**Ação**:
- Novo módulo `clustering.py` com **ONC** (silhouette + re‑seed + *upper‑level* search).
- `hrp.py`: usar ONC para definir **partição inicial** ou **leaf‑ordering** da árvore.

**Checklist**:
- Suporte a `X` como **matriz de observações** (ex.: distância de distâncias) ou `ρ` como insumo.
- Detonar antes de clusterizar (opcional).
- Relatório: *boxplots* de E[K]/K em dados sintéticos + estabilidade em dados reais.

---

### 2.4 Financial Labels (Cap. 5)
**Objetivo**: rótulos realistas para *meta‑models* (ex.: filtro de entrada/saída, regimes).  
**Ação**:
- Novo `labels.py` com **triple‑barrier** (PT/SL/tempo) e **trend‑scanning** (t‑score da inclinação).
- Integrar a `robustness.py` para testes de estacionariedade dos rótulos (z‑padronização por vol).

**Checklist**:
- Ajustar barreiras por **volatilidade prevista** (horizonte h).
- Evitar sazonalidade de *timestamp* → preferir **bars de dólar/volume** quando possível.
- Guardar *metadados* do rótulo (barreira tocada, duração, retorno realizado).

---

### 2.5 Feature Importance (Cap. 6)
**Objetivo**: avaliar sinais via **MDA** (permutation importance) e **MDI** de *ensembles*; evitar fetiche por **p‑values**.  
**Ação**:
- `kpis.py` ou novo `importance.py`: implementar **MDA cross‑validated** (com **Purged/Embargoed CV**).  
- Reportar **intervalos** e **substituibilidade** (colinearidade) — cuidado com *underestimation* em features correlacionadas.

**Checklist**:
- MDA com métrica apropriada (AUC/LogLoss para classificadores, não *accuracy*).  
- Log de sementes e *folds* para reprodutibilidade.

---

### 2.6 Portfolio Construction (Cap. 7)
**Takeaway**: média‑variância é instável → **HRP/T‑HRP** ganha com **(i) denoise/detone**, **(ii) distância robusta**, **(iii) clusterização estável**.  
**Ação**:
- No `engine.py`, permita *switch* entre **HRP padrão** e **T‑HRP** com artefatos acima.
- Em `risk_controls.py`, aplicar **turnover cap**, **kill‑switch**, **vol targeting** já existentes, e medir **RC by cluster**.

---

### 2.7 Test‑Set Overfitting (Cap. 8)
**Objetivo**: **CPCV** (Combinatorial Purged CV) + **deflated Sharpe** para validar hiperparâmetros e *model selection*.  
**Ação**:
- Estender `purged_cv.py` para modo **CPCV** (subconjuntos combinatórios sem *leakage* + *embargo*).
- `kpis.py`: incluir **deflated Sharpe** para penalizar busca extensiva.

**Checklist**:
- Registrar **nº de trials** por *search*, usar *early‑stopping*, e *reporting* honesto dos graus de liberdade.


---

## 3) Receitas Plug‑and‑Play (combináveis)

### R1) Pré‑processamento para T‑HRP
1. Calcular `corr` de retornos padronizados (janelas 252/504).  
2. **Denoise/Detone** (`preprocess_corr`).  
3. Medir **distância** `D` (VI ou √(0.5*(1-ρ))).  
4. **ONC** para K ótimo → clusters/árvore.  
5. **HRP** com pesos por *cluster risk parity*; aplicar `turnover_cap` e `kill_switch`.

### R2) Meta‑labeling de regime
1. Gerar rótulos **trend‑scanning** (up/flat/down).  
2. Features de *market state* (vol, breadth, TDA score).  
3. Classificador com **MDA** para importância.  
4. Decidir **alpha‑mix** (α,β,γ) em `factors.get_alphas_from_cfg` condicionado ao regime.

### R3) Validação robusta
1. **Purged K‑Fold** para *tuning* de TDA/Kepler.  
2. **CPCV** para seleção de configuração final (reportar deflated Sharpe).


---

## 4) Tarefas de implementação (Checklist Dev)

- [ ] `hrp.py`: `preprocess_corr()` com MP‑fit + detone + projeção.  
- [ ] `metrics.py`: MI/VI k‑NN + utilitários de discretização.  
- [ ] `clustering.py`: ONC (silhouette, multi‑seed, upper‑level).  
- [ ] `labels.py`: triple‑barrier + trend‑scanning (com *vol‑adjust*).  
- [ ] `importance.py`: MDA com Purged/Embargoed CV.  
- [ ] `purged_cv.py`: modo CPCV + util `embargo_indexer`.  
- [ ] `kpis.py`: deflated Sharpe e relatórios de estabilidade de pesos.  
- [ ] `engine.py`: *switches* de experimento e *logging* estruturado.


---

## 5) Notas de Contexto (para “dieta” do GPT)

- **Priorize**: Cap. 2, 3, 4, 5, 6, 8.  
- **Trechos para citar** (paráfrases no código/comentários):  
  - Marčenko–Pastur para denoise; detonar para remover fator comum.
  - Triple‑barrier e trend‑scanning para rótulos alinhados à gestão de posição.
  - ONC para recuperar K sem *guesswork*.
  - MDA/MDI > p‑values em *feature importance*.
  - CPCV + deflated Sharpe para validar.

---

## 6) Snippets de Config (exemplo)

```yaml
# base.yaml (trecho)
preprocessing:
  denoise: true
  detone: true
  mp_fit:
    method: "KDE"        # ou "analytic"
    min_eig: 1e-8

distance:
  metric: "vi"           # "1-corr" | "mi" | "vi"

clustering:
  method: "onc"
  max_k: 20
  reseeds: 16

labels:
  scheme: "trend_scanning"  # "triple_barrier"
  h_max_bars: 126
  vol_model: "ewm_63"

validation:
  cv: "cpcv"             # "purged_kfold"
  embargo_frac: 0.01

portfolio:
  constructor: "t_hrp"
  turnover_cap: 0.25
  vol_target: 0.10
```

---

## 7) Leituras rápidas (capítulos e porquê)
- **Cap. 2**: separa sinal/ruído em ρ/Σ; base para pesos estáveis.  
- **Cap. 3**: MI/VI capturam **não-linearidade** entre ativos/fatores.  
- **Cap. 4**: ONC automatiza K → melhora hierarquia do HRP.  
- **Cap. 5**: rótulos condizentes com **gestão de posição** (não só horizonte fixo).  
- **Cap. 6**: importância fora‑da‑amostra (**MDA**) > explicação in‑sample.  
- **Cap. 8**: proteção contra **overfitting de teste** (CPCV/deflated Sharpe).

---

## 8) Avisos práticos
- **Dados curtos/alta‑dimensão** → preferir *regularização* + hierarquia.  
- **Relatar incerteza** (intervalos, *bootstrap* de pesos, estabilidade de clusters).  
- **Controle de versões** de *seeds* e *folds*; evite *leakage* em toda a pipeline.

---

## 9) Próximos passos sugeridos
1) Implementar `preprocess_corr()` e comparar HRP vs T‑HRP com/sem denoise.  
2) Ligar ONC e capturar *diagnósticos* (E[K]/K em sintéticos).  
3) Integrar `labels.py` + `importance.py` e rodar MDA sobre *market‑state* features.  
4) Migrar validação para **CPCV** e adicionar **deflated Sharpe** nos relatórios.

---

**FIM — ML4AM para Atlas**
