from copy import deepcopy
import pandas as pd
from dataio.config import load_config
from src.main import _load_panel
from backtest.engine import run_backtest

def run_once(cfg_patch: dict, label: str):
    cfg = deepcopy(load_config("configs/base.yaml"))
    # merge raso (dicts aninhados ganham override)
    for k,v in cfg_patch.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    panel = _load_panel(cfg)
    out = run_backtest(cfg, panel=panel)
    print(f"\n=== {label} ===")
    print("KPIs:", out["kpis"])
    print("META.tda_params:", out["meta"].get("tda_params"))
    print("META.tfi_stats:", out["meta"].get("tfi_stats"))
    print("META.capacity:", out["meta"].get("capacity"))
    # utilidades de inspeção rápida
    weights = out.get("daily_positions")
    if isinstance(weights, pd.DataFrame):
        nnz = (weights.abs()>1e-12).sum(axis=1).mean()
        print("Média #ativos alocados/dia:", round(nnz,2))
    trades = out.get("trades")
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        print("Fees/slip totais:",
              float(trades["fees"].sum()), float(trades["slip"].sum()))
    return out

# EXEMPLOS DE TESTE (ative um por vez):

# A) TDA.window grande vs pequeno
out1 = run_once({"tda": {"window": 252}}, "TDA.window=252")
out2 = run_once({"tda": {"window": 84}},  "TDA.window=84")

# B) Vol target baixo vs alto
out1 = run_once({"vol_target": 0.08}, "vol_target=8%")
out2 = run_once({"vol_target": 0.20}, "vol_target=20%")

# C) Custos altos vs baixos
out1 = run_once({"costs": {"fee_bps": 2, "k": 0.05, "max_bps": 25}}, "custos baixos")
out2 = run_once({"costs": {"fee_bps": 20, "k": 0.5, "max_bps": 100}}, "custos altos")

# D) Universo enxuto vs largo
out1 = run_once({"universe": {"top_n": 5, "adv_min": 0, "price_min":0, "age_min": 20}}, "top_n=5")
out2 = run_once({"universe": {"top_n": 20}}, "top_n=20")

# E) Caps de participação/turnover
out1 = run_once({"participation_cap": 0.05, "turnover_cap": 0.10}, "caps duros")
out2 = run_once({"participation_cap": 0.20, "turnover_cap": 0.50}, "caps frouxos")

# F) Pesos de fatores (alphas)
out1 = run_once({"factors": {"alphas": [0.9, 0.1, 0.0]}}, "alphas=[0.9,0.1,0.0]")
out2 = run_once({"factors": {"alphas": [0.6, 0.4, 0.2]}}, "alphas=[0.6,0.4,0.2]")

# G) Softmax_T da mistura cross-section
out1 = run_once({"factors": {"softmax_T": 0.3}}, "softmax_T=0.3 (mais concentrado)")
out2 = run_once({"factors": {"softmax_T": 2.0}}, "softmax_T=2.0 (mais difuso)")

# tests = [
#     ({"participation_cap": 0.5, "turnover_cap": 1.0}, "caps_frouxos"),
#     ({"universe": {"top_n": 30}}, "top_n=30"),
#     ({"vol_target": 0.08}, "vol_target=8%"),
#     ({"participation_cap": 0.5, "turnover_cap": 1.0, "universe": {"top_n": 30}}, "combo_caps+topn"),
# ]
# for patch, label in tests:
#     out = run_once(patch, label)
#     print("cap_bind_rate:", out["meta"]["capacity"]["cap_bind_rate"],
#           "| avg_turnover_cut_frac:", out["meta"]["capacity"]["avg_turnover_cut_frac"])
    
# w = out["daily_positions"]
# cap = out["meta"]["capacity"]["participation_cap"] if "capacity" in out["meta"] else 0.5
# hit_rate = ((w.abs() >= cap - 1e-12).sum(axis=1) / w.shape[1]).mean()
# print("Frac. média de ativos no cap:", round(hit_rate,3))

# out = run_once({}, "baseline_patched")
# print(out["meta"]["capacity"])
# print("annual_vol:", out["kpis"]["annual_vol"], "| vol_target:", 0.12)

# # checar se turnover não muda a escala (soma antes/depois)
# prev = out["daily_positions"].iloc[-2]
# tgt  = out["daily_positions"].iloc[-1]
# print("sum(prev), sum(tgt):", float(prev.sum()), float(tgt.sum()))
