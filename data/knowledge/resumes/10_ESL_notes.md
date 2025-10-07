# The Elements of Statistical Learning (ESL) — Notas de estudo
**Referência:** Trevor Hastie, Robert Tibshirani, Jerome Friedman — *The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2ª ed., 2009)*.

> **TL;DR**: ESL é o “mapa” dos métodos de aprendizado estatístico. Cobrem-se fundamentos (viés–variância, avaliação via CV/bootstrapping), métodos lineares e não lineares (ridge, lasso, splines, kernels), árvores/boosting/random forests, SVMs, GMs e cenários de alta dimensionalidade (*p ≫ n*). A mensagem central: **modelos simples bem validados vencem a maioria dos truques**, e regularização + boa avaliação superam ajustes sofisticados com validação ruim.

---

## 1) Ideias-chave (para quant)
- **Viés–variância**: erro esperado = viés² + variância + ruído. Aumentar a flexibilidade tende a reduzir viés e elevar variância; regularização faz o inverso (Cap. 2, 7).
- **Regularização como prior implícito**: penalizar complexidade (L2, L1, TV/fused) estabiliza parâmetros, aumenta robustez e melhora generalização (Cap. 3, 5, 18).
- **Seleção de modelo é parte do modelo**: usar **CV** (k-fold, LOOCV) ou **bootstrap** para estimar risco fora da amostra (Cap. 7–8). Separar *tuning* de *assessment*.
- **Métodos de conjunto**: *bagging* reduz variância (árvores instáveis) e **boosting** segue descida de gradiente em espaço de funções com perda aditiva (Cap. 10, 15–16).
- **Kernels e espaços RKHS**: mapear implicitamente para alta dimensão e aplicar regularização (SVM, kernel ridge, kernel PCA). Escolha de kernel/banda = controle de complexidade (Cap. 5, 6, 12, 14).
- **Alta dimensionalidade (*p≫n*)**: induz sobreajuste, instabilidade e “esparsidade benigna”: lasso, Dantzig, shrinkage diagonal, seleção múltipla com FDR (Cap. 18).
- **Interpretação**: importância de variáveis (RF, GBM), *partial dependence*, suavizações aditivas (GAM) e gráficos diagnósticos (Cap. 9, 10, 15).

---

## 2) Equações essenciais (mínimo operacional)
- **Ridge (L2)**:  \(\hat\beta = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2\). Fecha com \(\hat\beta=(X^TX+\lambda I)^{-1}X^Ty\).
- **Lasso (L1)**:  \(\hat\beta = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1\). Esparsidade; solução por LARS/coordinate descent.
- **Logística**:  \(\ell(\beta)=\sum_i [y_i x_i^T\beta-\log(1+e^{x_i^T\beta})]-\text{pen}(\beta)\). *Pen* pode ser L1/L2.
- **SVM (soft-margin, primal)**:  \(\min_{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum_i \xi_i\) s.a. \(y_i(w^T\phi(x_i)+b)\ge 1-\xi_i,\ \xi_i\ge0\). Kernel via \(K(x,x')=\phi(x)^T\phi(x')\).
- **Boosting como otimização**: itera \(f_{m+1}=f_m+\nu \gamma_m h_m\) escolhendo base learner \(h_m\) para descer gradiente da perda (ex.: exponencial, deviance) (Cap. 10).
- **Árvore CART**: partições recursivas pelo ganho de impureza (Gini/entropia/MSE), *cost-complexity pruning* \(R_\alpha(T)=R(T)+\alpha|T|\) (Cap. 9).
- **Random Forests**: *bagging* + *feature subsampling* em cada *split*; OOB para erro/VI (Cap. 15).
- **Bias–variance (regressão)**: \(\mathbb E[(Y-\hat f(X))^2]=\sigma^2 + \text{Bias}^2 + \text{Var}\) (Cap. 7).
- **FDR (Benjamini–Hochberg)**: controla \(\mathrm{FDR}\) sobre muitos testes; ordenar *p-values* e definir *cutoff* \(p_{(k)}\le \frac{k}{p}q\) (Cap. 18).
- **EM**: alterna *E-step* (Q) e *M-step* para máximizar verossimilhança com latentes (Cap. 8).
- **BIC/MDL**: penalizam dimensão efetiva; aproximam risco fora da amostra (Cap. 7).

---

## 3) Resumo guiado por capítulos (ultracondensado)
- **2** Supervisionado: regressão vs. classificação; k-NN vs. modelagem linear; *curse of dimensionality*.
- **3** Regressão linear: subset/forward/backward, ridge, lasso, LARS, PCR/PLS; *path algorithms*.
- **4** Classificação linear: LDA/QDA, logística (com L1/L2), separadores ótimos, *perceptron*.
- **5–6** *Basis* e suavização: splines, smoothing splines, wavelets; kernel smoothing/density; *bandwidth*.
- **7** Avaliação: *optimism*, Cp/AIC/BIC/MDL, VC, **cross-validation** (jeito certo), *bootstrap*.
- **8** Inferência & *ensembles*: *bootstrap* vs. ML, Bayes, **EM**, MCMC, **bagging**, stacking/bumping.
- **9** GAM, árvores, MARS, PRIM; *missing data*.
- **10** **Boosting** (AdaBoost, gradient boosting), perdas robustas, shrinkage, *subsampling*, interpretação (PDP/importance).
- **11** Redes neurais (MLP): treino, *overfitting*, penalização, *Bayesian NNs*.
- **12** **SVMs** e discriminantes flexíveis (FDA, PDA, MDA; SVR).
- **13** *Prototypes & kNN*: k-means, LVQ, métricas invariantes, redução para kNN.
- **14** Não supervisionado: regras de associação, *clustering* (k-means/GMM/hierárquico), PCA/KernelPCA/SparsePCA, NMF, ICA, MDS, *spectral clustering*, PageRank.
- **15** **Random Forests**: definição, OOB, VI, *proximities*.
- **16** Ensembles: *regularization paths*, *bet on sparsity*, *rule ensembles*.
- **17** Modelos gráficos não dirigidos: grafos de Markov (contínuos/discretos), estrutura e parâmetros.
- **18** **p≫n**: discriminação diagonal, shrinkage quadrático e L1 (lasso/fused), *string kernels*, *SPC*, múltiplos testes e **FDR**.

---

## 4) Padrões práticos (o que o ESL recomenda na essência)
1) **Pipeline de validação**: *train/validation/test* ou **CV aninhada** para *tuning* + avaliação final.
2) **Regularize sempre**: prefira *paths* (λ) e escolha por CV; em *p≫n*, L1/L2 híbridos e *screening* inicial.
3) **Ensembles quando base learner é instável** (árvores): bagging/RF; quando precisa de *fit* forte: boosting (com *shrinkage* pequeno e *subsampling*).
4) **Medidas fora da amostra**: OOB (RF), CV, *bootstrap* .632+; jamais use erro de treino.
5) **Inspecione**: curvas CV vs. complexidade, importâncias, PDP/ICE, resíduos e *leverage*.
6) **Escolha da perda**: quadrática (reg.), log-loss (class.), Huber/quantil (robustos).

---

## 5) Riscos comuns & como mitigar
- **Vazamento de informação**: *feature scaling/selection* deve ocorrer **dentro** de CV; use *pipelines*.
- **Overfitting em *tuning***: CV aninhada; *early stopping* (boosting/NN).
- **Dados desbalanceados**: *class weights*, amostragem estratificada, métricas apropriadas (PR-AUC).
- **Alta correlação entre features**: ridge/elastic-net; atenção à instabilidade de importância.
- **p≫n e múltiplos testes**: controle **FDR**, *stability selection*.
- **Extrapolação**: métodos locais (kNN/kernels) degradam fora do *support*; monitorar *distance-to-train*.
- **Custo computacional**: kNN/Kernel exigem estruturas de vizinhança/approx; árvores/ensembles escalam melhor com *n*.

---

## 6) Checklist de aplicação (rápido)
- Defina **perda** e métrica de negócio.
- Faça **split** com estratificação/temporalidade quando necessário; congele *test*.
- Construa **pipeline**: *preprocess* → *model* → *metric*. Toda lógica **dentro** de CV.
- Faça *tuning* parcimonioso (grid/bayes) e **pare cedo**. Salve curva de validação.
- Audite estabilidade (semente, *folds*, *perturbações*). Gere **card de risco** do modelo.
- Documente *features*, *tuning*, *seed*, *data ranges*, *data checks* e resultado OOS.

---

## 7) Mapeamento → Atlas (T‑HRP v3.0)
**Onde ESL encaixa no fluxo do Atlas (estratégia topológica com gestão de risco ativa):**
- **Geração/seleção de *features*** (`factors.py`): PCA/KernelPCA/SparsePCA (Cap. 14, 18); lasso/elastic-net para *screening* (Cap. 3, 18).
- **Modelagem de sinais** (`engine.py`): modelos lineares regularizados, árvores/GBM, SVM; perdas adequadas ao alvo (regressão de retornos vs. classificação de direções).
- **Validação temporal** (`purged_cv.py`, `walk_forward.py`): princípios do Cap. 7 (CV correta) + *purged k-fold/embargo* para séries (extensão prática).
- **Robustez** (`robustness.py`): *stress* por *resampling*, *noise injection*, *feature dropout* (Cap. 8 *bagging* como *variance reduction*).
- **Dimensionamento & risco** (`risk_controls.py`, `hrp.py`): usar importâncias/PDP para *guardrails*; preferir sinais estáveis (baixo turnover).
- **Custo & execução** (`costs.py`, `execution.py`): escolher modelos que maximizem **IR neto**, não só acurácia bruta; *shrinkage* ajuda a reduzir *turnover*.

> **Padrão recomendado no Atlas**: para sinais *p≫n*, começar por **elastic‑net** ou **GBM com *shrinkage***, validar com **walk‑forward purged**, e selecionar hiperparâmetros pelo **critério de menor complexidade dentro de 1‑SE** na curva de CV.

---

## 8) “Quando usar o quê” (heurística rápida)
- **Lineares (ridge/elastic‑net)**: forte baseline, rápidos, bons com *p≫n* e colinearidade.
- **Árvores/RF**: interações automáticas, robustos, baixa necessidade de *scaling*, métricas OOB.
- **Boosting (GBM/XGB‑like)**: melhor *fit* com *shrinkage* pequeno; cuidado com *overfitting*; ótimo para *tabular*.
- **SVM**: margens claras e *kernels* bem escolhidos; podem ser pesados em *n* grande.
- **kNN/kernels locais**: bons localmente; evite em *d* alto sem redução de dimensão.
- **GAM**: interpretáveis, não lineares suaves; ótimos como *glass box*.

---

## 9) Referências cruzadas úteis (do próprio livro)
- Cap. **7** (CV “do jeito certo”), **10** (boosting = gradiente), **15** (OOB/VI), **18** (FDR, *p≫n*).

---

### Anexo A — Exemplos de *loss* por objetivo
- Retorno direto (reg.): MSE/Huber/Quantil (τ=0.5 ou 0.6).
- Direção (class.): Log-loss; métricas OOS: AUC/PR‑AUC, *balanced accuracy*.
- Ranqueamento/seleção: *pairwise loss* (hinge/logística).

### Anexo B — Pseudocódigo de validação (CV aninhada)
```
for fold_out in K_out:
    split train_in, valid_in from train_out
    for λ in grid:
        score[λ] = CV_in(model(λ), train_in, valid_in)
    λ* = argmin_λ  mean(score[λ])  (usar 1-SE rule)
    fit_on_full_train_out(model(λ*))
    eval_on_holdout(fold_out.test)
report mean ± se over K_out
```

---

> **Resumo executivo**: **regularize, valide direito, monitore estabilidade**. Ensembles (bagging/boosting/RF) são poderosos quando combinados com perda adequada e *early stopping*. Em *p≫n*, esparsidade e controle de FDR são essenciais. Esses princípios alinham com a filosofia do Atlas (T‑HRP): **sinais estáveis + avaliação honesta + gestão de risco ativa**.
