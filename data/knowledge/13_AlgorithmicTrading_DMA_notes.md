# Algorithmic Trading & Direct Market Access — Notas de estudo
**Referência:** Barry Johnson — *Algorithmic Trading and Direct Market Access* (2009).

> **TL;DR:** O livro é o manual de **execução eletrônica** e **microestrutura**: como a ordem atravessa gateways DMA/SOR até o *matching engine*; quais táticas (VWAP/TWAP/POV/IS) usar; como medir **slippage** e **impacto**; e que **controles de risco/latência** são obrigatórios. A mensagem central: **custos de execução dominam** a performance de muitas estratégias; trate execução como parte do *alpha*.

---

## 1) Microestrutura & acesso (o terreno de jogo)
- **Livro de ofertas (LOB):** *price–time priority*, *tick size*, profundidade por nível, *queue position*.  
- **Fragmentação & roteamento:** múltiplos *venues*, *smart order routing (SOR)*, *maker–taker* (rebates/fees).  
- **Leilões & pausas:** abertura/fechamento, *volatility auctions*, *circuit breakers*.  
- **Ordens & flags:** *market/limit/stop*, *IOC/FOK*, *pegged*, *iceberg/hidden*, midpoint.  
- **DMA vs. *sponsored access*:** latência, *risk checks* pré‑negociação, *kill switch*, *drop copy* e auditoria.  
- **Latência:** *colocation*, *gateway threading*, *batching*, *throttling*, *clock sync* (NTP/PTP).

---

## 2) Táticas de execução (como fatiar e quando)
- **TWAP** (time‑weighted): fatia por tempo; robusta e simples quando volume é uniforme.  
- **VWAP** (volume‑weighted): segue perfil histórico/intradiário; **dinâmica** quando usa volume *real‑time*.  
- **POV** (*participation of volume*): mantém fração alvo \(\rho\) do *tape*.  
- **IS / Arrival Price** (*implementation shortfall*): minimiza desvio desde o preço de decisão; equilibra **impacto × risco de preço**.  
- **Leilões**: bons para blocos e *rebalance*; risco de *gaming*.  
- **Child order placement:** gerenciar *queue*, *price stepping*, *pegging* ao *mid*, *join/lean* nas melhores ofertas.

---

## 3) Medindo custos: decomposição & modelos
- **Slippage total** = **spread** + **impacto** (temporário/permanente) + **atraso** + **taxas/rebates**.  
- **Implementation Shortfall (bps):**  
  \[ \text{IS} = \operatorname{side}\cdot\frac{\sum_i q_i p_i - Q\,p_0}{Q\,p_0}\times 10^4 \]
  onde \(p_0\) é o **arrival**, \(q_i,p_i\) execuções parciais e \(Q=\sum q_i\).  
- **Regra da raiz quadrada (impacto esperado):**  
  \[ I \approx Y\,\sigma\,\sqrt{\tfrac{Q}{V}} \] 
  com **vol intradiária** \(\sigma\), tamanho \(Q\), volume de mercado \(V\) e **constante** \(Y\sim 0.5\text{–}1.0\).  
- **Capacidade:** custo cresce com \(Q/V\); **participation caps** evitam entrar no regime não linear.  
- **Seleção de *venue***: compare **fill rate**, **adverse selection** (P&L pós‑trade), **rebates** e latência.

---

## 4) Arquitetura DMA/SOR (de fora para dentro)
- **FIX/API → Gateways** (normalização, validações, *pre‑trade risk*).  
- **Regras de risco**: *fat finger* (tamanho/preço), *price collars*, limites de notional, *max order rate*, **net exposure**.  
- **Roteador (SOR):** lógica de *spray/sequencing*, *pegging*, *dark/visible sweep*, *queue prediction*, *anti‑gaming*.  
- **State & telemetria:** livro consolidado, *drop copy*, *heartbeat*, *replay*, logs por **ordem/exec** para TCA e auditoria.  
- **Matching engine:** *price–time* ou *pro‑rata*; *self‑trade prevention*; *auction uncrossing*.

---

## 5) Regras de mercado e efeitos práticos
- **Tick size** grande → filas menores e *price stepping* mais caro; pequeno → mais micro‑oscilações.  
- **Maker–taker:** *routing* sensível a rebates; monitore **P&L líquido de taxas**.  
- **Dark pools/midpoint:** menos impacto, risco de seleção adversa; *minimum quantity* e *last‑look* importam.  
- **Pausas/haltes:** algoritmos devem **reagir a leilões** e “cool‑downs” (pausar POV/VWAP e replanejar).

---

## 6) TCA (Transaction Cost Analysis) que importa
- **Benchmarks:** *arrival*, *decision*, *open/close*, **VWAP**.  
- **Métricas:** IS (bps), *fill rate/timeout*, *spread capture*, **realized spread**, *mark‑out* (1–5 min), **impact vs. timing**.  
- **Estratos:** por *cap*, volatilidade, *participation*, *venue*; **cartas de controle** para desvios.  
- **Uso**: *feedback loop* p/ perfilar táticas, *venue lists* e parâmetros (\(\rho\), *urgency*).

---

## 7) Crosswalk → Atlas (projeto T‑HRP v3.0)
- **`execution.py`**: implementar **POV, VWAP (dinâmico), TWAP e IS**; *scheduler* com *urgency* e **participation caps**.  
- **`costs.py`**: incluir **spread dinâmico**, **rebates**, e **impacto raiz‑quadrada**; parametrizar por **regime** de vol/liquidez.  
- **`capacity.py`**: curvas de Sharpe/retorno **vs participação** (ligar ao \(Q/V\)); *what‑if* por *caps*.  
- **`risk_controls.py`**: *pre‑trade checks*, **kill switch**, *exposure caps* por ativo/cluster; pausas em leilões.  
- **Telemetria**: *drop‑copy* sintética (logs por *child order*); KPIs de **fill**, **mark‑out**, **impact share**.

---

## 8) Pseudocódigos úteis
**POV (participação \(\rho\))**
```
for each slice window:
    traded = market_volume(window) * rho - executed_so_far
    price  = choose_quote(bid/ask, peg, step)
    post_or_cross(traded, price, ioc=True)
```
**VWAP dinâmico**
```
target = forecast_profile(t) * Q_total
need   = max(target - executed, 0)
send child orders with price/pegging to minimize mark-out
```
**IS (chegada)**
```
arrival = mid_price_at_start()
while executed < Q:
    decide urgency from (vol, spread, drift, time_left)
    send child orders; update cumulative IS and risk
```

---

## 9) Armadilhas & boas práticas
- **Perfil VWAP estático em regime volátil** → derrapa; prefira perfil **adaptativo**.  
- **Ignorar fees/rebates** → *routing* subótimo.  
- **Sem *queue management*** → *timeouts* e *toxicity* altos; use *pegging* e *join/lean*.  
- **Participação alta** sem *caps* → impacto explode.  
- **Logs fracos** → sem TCA, sem melhoria; **telemetria detalhada** é essencial.

---

## 10) Ligações com outros notes do `/knowledge`
- **08_AlmgrenChriss_notes**: modelo de **trade‑off impacto×risco** para *scheduling*.  
- **09_Hull_Risk_notes**: liquidez como risco; *stress* e **RAROC**.  
- **04_APM_notes**: **TC** (transfer coefficient) cai com restrições/custos → execução afeta IR.  
- **11_Hilpisch_notes**: *event‑driven backtest* e engenharia de execução em Python.

---

> **Resumo executivo:** Trate execução como *primeira‑classe*: selecione tática/venue por **vol/liquidez/urgência**, imponha **caps**, meça com **TCA**, e feche o *loop* ajustando parâmetros (\(\rho\), *pegging*, *queues*) conforme o regime. Assim o **alpha líquido** sobe e a estratégia fica **escalável**.
