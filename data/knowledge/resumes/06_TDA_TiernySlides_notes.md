# TDA (Tierny) — Notas para o Atlas (T‑HRP)

tags: [tda, persistence, reeb-graph, contour-tree, morse-smale, discrete-morse, pl, scalar-field, ttk]

## 1) Tese (80/20)
A **Topological Data Analysis (TDA)** resume a “forma” de um campo escalar em poucas estruturas **estáveis, hierárquicas e interpretáveis**:
- Representar os dados como **campo escalar PL** (piecewise linear) sobre uma **triangulação/PL-manifold**;
- Extrair **pontos críticos** (mín., selas, máx.) e **pareá‑los** (nascimento/morte) → **persistência** separa **sinal vs. ruído** (estável à norma ∞);
- Contrair componentes de **nível** → **Reeb graph / Contour tree** (adjacências por nível);
- Usar **linhas integrais** (∇f) → **Morse‑Smale complex** (partição canônica por origem/destino), também com versão **Discrete Morse** (robusta).

Resultado prático: **segmentação** e **medidas** (contagem, área, “lifetime” de feições) que mudam **pouco** sob ruído e servem de **sinais**/regimes.

---

## 2) Conceitos essenciais (mínimo operacional)
**Domínio e faixa (range)**  
- Dados em malhas regulares/irregulares → **triangulação** (complexo simplicial); opção preferida: **PL‑manifold**.  
- **Campo escalar PL**: valores nos vértices (\(\hat f\)), interpolação **linear** no simplex → \(\nabla f\) é **constante por simplex**.  
- **Empate de valores**: tornar \(\hat f\) **injetiva** (p. ex., *simulation of simplicity* com desambig. por ordem dos vértices).

**Conjuntos de nível**  
- **Sub/super-level**: \(L^-(i)=\{p|f(p)\le i\}\), **level set**: \(f^{-1}(i)\) (\((d-1)\)-variedade no caso regular).  
- **Lower/upper link** definem regularidade/criticidade local do vértice.

**Críticos (PL‑Morse)**  
- Vértice é **regular** se **lower** e **upper links** são simplesmente conexos; senão é **crítico**.  
- Em 2D: índices 0/1/2 = mín./sela/máx.; em 3D: 0, 1‑sela, 2‑sela, 3 (máx.).  
- **Morse‑Euler**: contagens de críticos respeitam \(\chi(M)\).

**Persistência**  
- Construir **filtração** por subníveis; parear críticos (regra do **Elder**);  
- **Diagrama de Persistência**: pontos (nasc., morte); **Curva de Persistência**: #pares com persistência \(>\varepsilon\) (útil para threshold).  
- **Estabilidade**: pequenas perturbações em \(f\) ⇒ pequenos desvios no diagrama (métrica *bottleneck*).

**Reeb / Contour Tree**  
- Contrai cada **componente de nível** em um nó → grafo 1D que captura **conexões e bifurcações** por nível.  
- Em domínios simplesmente conexos (\(\beta_1=0\)): vira **Contour Tree**. Facilita *seeding* ótimo de isosuperfícies e **segmentação** por nível.

**Morse‑Smale**  
- Usa **linhas integrais** (origem/destino críticos) para particionar o domínio em **células** (ascendentes/descendentes).  
- **Discrete Morse** evita degenerescências e escala bem; simplificação por persistência gera **hierarquias** de segmentações.

---

## 3) Algoritmos & práticas (TTK & cia.)
- **Críticos**: teste local de links (Banchoff). **Persistência**: redução esparsa (ELZ) ou via árvores (para pares extremum‑sela).  
- **Contour Tree / Reeb**: *join/split trees* com *union‑find* (tempo quase ótimo), versões multi‑thread (Contour Forests).  
- **Morse‑Smale (discreto)**: construir **gradiente discreto** (pareamento de células) → V‑paths; simplificar por persistência.  
- **Simplificação topológica**: remover pares com persistência \(<\varepsilon\) mantendo \(\|f-g\|_\infty\le\varepsilon\).  
- **Ferramentas**: **Topology ToolKit (TTK)**, bindings Python/C++, e pipelines interativos para extrair/simplificar/segmentar.

---

## 4) Como isso vira sinal no Atlas (T‑HRP)
**Objetivo**: extrair um **score de regime/fragilidade topológica (TFI)** por ativo/janela e features para **seriação** (ordem HRP) e **risk controls**.

1) **Pré‑processo (janela W)**  
   - Normalizar retornos; suavisar opcional (EMA curta). Garantir **injetividade** de \(\hat f\).

2) **Abstrações & métricas** (por ativo)  
   - **Persistência**: área média do diagrama, soma de *lifetimes*, #pares acima de \(\varepsilon\) (curva com *knee* automático).  
   - **Reeb/Contour tree**: #ramos ativos no intervalo, profundidade média, variação temporal de ramos.  
   - **Morse‑Smale**: contagem/estabilidade de células dominadas por máx. (trend) vs. mín. (reversão).

3) **TFI (0–1)**  
   - Agregar métricas (z‑score → *squash* logístico) e **suavizar** no tempo (EWMA).  
   - **Interpretação**: TFI alto = **paisagem mais irregular** (muitos eventos topológicos persistentes) ⇒ regime “arriscado/caótico”.

4) **Integração com HRP**  
   - **Seriação**: usar hierarquia (ordenar por similaridade topológica ou usar ordem TDA como *leaf ordering*).  
   - **Risco**: mapear TFI → **target vol/gross** (regime‑aware), **caps** de participação/turnover, histerese de *killswitch*.

> **Nota**: No repositório Atlas, `tda.py` expõe funções de TDA; `factors.py` computa `tfi_score` (normalizado 0–1); `engine.py` usa `regime_value` para target‑vol/gross; `hrp.py` aceita seriação TDA para quase‑diagonalizar Σ antes da bisseção.

---

## 5) Dials que mais importam
- **Janela (W)** e **suavização** (EMA): curto = responsivo; longo = estável.  
- **Threshold de persistência (\(\varepsilon\))**: *Otsu/kneedle/plateau* na **curva de persistência**; validar por *walk‑forward*.  
- **Escolha da abstração**: Reeb/Contour para conectividade por nível; Morse‑Smale para bordas guiadas por gradiente.  
- **Granularidade da malha**: *downsample* adaptativo mantém topologia (evita aliasing).  
- **Regularização**: *jitter* injetivo; filtros morfológicos leves se ruído de alta frequência contamina pares de baixa persistência.

---

## 6) Pitfalls & mitigação
- **Degenerescências (empates/planos)** → *simulation of simplicity* e/ou *Discrete Morse*.  
- **Over‑segmentation** por ruído → **threshold de persistência** + *smoothing* leve.  
- **Custo** em 3D/alto |simplices| → usar **Contour Tree** (loop‑free), algoritmos *output‑sensitive* e *caching*.  
- **Sensibilidade a bordas** → tratar **boundary components** e comparar com domínio expandido (*padding*) quando viável.  
- **Mudança de regime falsa** (saltos breves) → **histerese**/*median filter* no TFI.

---

## 7) Checklist de validação no Atlas
- [ ] TFI correlaciona com **aumento de vol/MDD** e **queda de IC HRP**.  
- [ ] Threshold \(\varepsilon\) escolhido nos **plateaus** da curva de persistência.  
- [ ] Estabilidade OOS das contagens (#ramos/#células) sob *bootstraps*.  
- [ ] Seriação TDA **reduz turnover** vs. seriação aleatória e melhora **quase‑diagonalização** de Σ.  
- [ ] Integração regime→**target vol/gross** melhora **Calmar** sem inflar custos/capacidade.

---

## 8) Referências úteis (curtas)
- **Tierny — Introduction to Topological Data Analysis** (apostila/slidebook).  
- **Edelsbrunner & Harer — Computational Topology** (cap. Persistência).  
- **TTK** (Topology ToolKit) — implementações de Reeb/Contour/Morse‑Smale e simplificação por persistência.
