from __future__ import annotations

"""Core backtesting engine scaffolding."""

import os
from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from backtest.execution import execute_trade
from dataio.loaders import get_adv, get_panel, select_universe
from features import TFIParams, mix_scores, momentum_12_1, quality_proxy, tfi_score, get_alphas_from_cfg
from portfolio import hrp_weights_from_order, rolling_cov, topo_seriation_from_graph
from risk import atr, atr_risk_normalize, scale_to_vol
from risk import risk_controls

logger = logging.getLogger(__name__)

__all__ = ["run_backtest"]


@dataclass
class BacktestState:
    equity: float
    current_weights: pd.Series
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    vol_series: pd.Series
    weights_history: Dict[pd.Timestamp, pd.Series]
    trades: List[Dict[str, float]]


def run_backtest(cfg: Dict, panel: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Run a deterministic monthly-rebalanced HRP backtest."""

    seed = int(cfg.get("seeds", 42))
    np.random.seed(seed)

    dates_cfg = cfg.get("dates", {})
    start = pd.Timestamp(dates_cfg.get("start"))
    end = pd.Timestamp(dates_cfg.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Missing start/end dates in configuration")

    logger.info("Loading market data from %s to %s", start.date(), end.date())
    if panel is not None:
        panel_data = panel.sort_index()
    else:
        panel_data = get_panel(start, end)

    if not isinstance(panel_data.index, pd.MultiIndex):
        raise ValueError("panel must have a MultiIndex with a date level")

    date_level = (
        "date" if "date" in panel_data.index.names else panel_data.index.names[0]
    )
    date_index = panel_data.index.get_level_values(date_level)
    mask = (date_index >= start) & (date_index <= end)
    panel = panel_data.loc[mask]
    if panel.empty:
        raise ValueError("Panel slice is empty for requested date range")

    prices = panel["close"].unstack("asset").sort_index()

    returns = prices.pct_change()
    returns = returns.fillna(0.0)

    logger.info("Computing liquidity metrics and volatility estimates")
    adv_series = get_adv(panel)
    adv_df = adv_series.unstack("asset").reindex(prices.index)
    adv_notional = adv_df * prices

    windows_cfg = cfg.get("windows", {})
    atr_len = int(windows_cfg.get("atr_len", 14))
    vol_window = int(windows_cfg.get("vol_window", 60))

    atr_df = atr(prices, n=atr_len)
    cov_dict = rolling_cov(returns, window=vol_window)

    # logger.info("Pre-computing factor scores")
    # tfi_params = cfg.get("tda", {})
    # # epsilon pode ser None (modo adaptativo no tda.py)
    # _eps_raw = tfi_params.get("epsilon", 0.5)
    # if _eps_raw is None or str(_eps_raw).lower() in {"none", "null"}:
    #     _eps = None
    # else:
    #     _eps = float(_eps_raw)
    # tfi_cfg = TFIParams(
    #     delay=int(tfi_params.get("delay", 1)),
    #     dim=int(tfi_params.get("dim", 3)),
    #     n_cubes=int(tfi_params.get("n_cubes", 4)),
    #     overlap=float(tfi_params.get("overlap", 0.5)),
    #     epsilon=_eps,
    #     min_samples=int(tfi_params.get("min_samples", 3)),
    #     window=int(tfi_params.get("window", max(vol_window, 63))),
    # )
    logger.info("Pre-computing factor scores")
    # NOVO: constrói TFIParams a partir do YAML (tda.*, features.tfi.* ou topo)
    tfi_cfg = TFIParams.from_config(cfg, vol_window_fallback=vol_window)
    logger.info("TFI used: delay=%s dim=%s n_cubes=%s overlap=%s eps=%s window=%s",
                tfi_cfg.delay, tfi_cfg.dim, tfi_cfg.n_cubes, tfi_cfg.overlap,
                tfi_cfg.epsilon, tfi_cfg.window)
    alpha, beta, gamma = get_alphas_from_cfg(cfg)
    logger.info("Alphas used: alpha=%.3f beta=%.3f gamma=%.3f", alpha, beta, gamma)
    factors_cfg = cfg.get("factors", {}) or {}
    regime_gain_cfg = factors_cfg.get("regime_gain")
    regime_mode_cfg = factors_cfg.get("regime_mode")
    env_gain = os.getenv("REGIME_GAIN")
    env_mode = os.getenv("REGIME_MODE")
    gain_source = regime_gain_cfg if regime_gain_cfg is not None else env_gain
    try:
        regime_gain_effective = float(gain_source) if gain_source is not None else 1.0
    except (TypeError, ValueError):
        logger.warning("Invalid regime gain override '%s'; defaulting to 1.0", gain_source)
        regime_gain_effective = 1.0
    mode_source = regime_mode_cfg if regime_mode_cfg is not None else (env_mode or "tanh")
    regime_mode_effective = str(mode_source).lower()
    if regime_mode_effective not in {"tanh", "linear", "power"}:
        logger.warning("Invalid regime mode override '%s'; defaulting to 'tanh'", mode_source)
        regime_mode_effective = "tanh"
    logger.info(
        "Regime modifiers: gain=%.3f mode=%s (cfg=%s env=%s)",
        regime_gain_effective,
        regime_mode_effective,
        regime_gain_cfg,
        env_mode,
    )
    tfi_meta = {
        "delay": tfi_cfg.delay,
        "dim": tfi_cfg.dim,
        "n_cubes": tfi_cfg.n_cubes,
        "overlap": tfi_cfg.overlap,
        "epsilon": tfi_cfg.epsilon,
        "min_samples": tfi_cfg.min_samples,
        "window": tfi_cfg.window,
    }
    tfi_meta = {
        "delay": tfi_cfg.delay, "dim": tfi_cfg.dim, "n_cubes": tfi_cfg.n_cubes,
        "overlap": tfi_cfg.overlap, "epsilon": tfi_cfg.epsilon,
        "min_samples": tfi_cfg.min_samples, "window": tfi_cfg.window,
    }

    regime_series = (
        tfi_score(prices, params=tfi_cfg).reindex(prices.index).ffill().fillna(0.0)
    )
    # Estatísticas da série de regime para debug/heatmaps
    if regime_series.empty:
        tfi_stats = {"min": np.nan, "max": np.nan, "std": np.nan, "mean": np.nan}
        logger.warning("TFI regime series is empty after window=%d; mix_scores will receive zeros.", tfi_cfg.window)
    else:
        vals = regime_series.values.astype(float)
        tfi_stats = {
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "std": float(np.nanstd(vals)),
            "mean": float(np.nanmean(vals)),
        }
        logger.info("TFI regime stats: min=%.3f max=%.3f mean=%.3f std=%.3f", tfi_stats["min"], tfi_stats["max"], tfi_stats["mean"], tfi_stats["std"])
    ###

    #### Testes ####
    #print("Regime Series:\n", regime_series, "\n\n")
    #print("Qtd Nan:\n", (regime_series == 0.0).sum(), "\n\n")
    #### ------ ####

    momentum_df = momentum_12_1(prices).reindex(prices.index).ffill().fillna(0.0)
    quality_df = quality_proxy(prices).reindex(prices.index).ffill().fillna(0.0)

    # factors_cfg = cfg.get("factors", {})
    # alphas = factors_cfg.get("alphas", [0.6, 0.3, 0.1])
    # if alphas and isinstance(alphas[0], (int, float)):
    #     alpha, beta, gamma = (list(alphas) + [0.3, 0.1])[:3]
    # else:
    #     alpha, beta, gamma = 0.6, 0.3, 0.1


    common_index = regime_series.index.intersection(momentum_df.index).intersection(
        quality_df.index
    )
    if common_index.empty:
        mix_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    else:
        regime_series = regime_series.reindex(common_index)
        momentum_df = momentum_df.reindex(common_index)
        quality_df = quality_df.reindex(common_index)
        mix_df = mix_scores(
            regime_series,
            momentum_df,
            quality_df,
            alpha,
            beta,
            gamma,
            regime_gain=regime_gain_effective,
            regime_mode=regime_mode_effective,
        )
        mix_df = mix_df.rename(columns=lambda c: c.replace("mix_", ""))
        mix_df = mix_df.reindex(prices.index).ffill().fillna(0.0)

    turnover_cap = float(cfg.get("turnover_cap", 0.25))
    target_vol = float(cfg.get("vol_target", 0.10))
    fee_bps = float(cfg.get("costs", {}).get("fee_bps", 5.0))
    slip_params = {
        "k": float(cfg.get("costs", {}).get("k", 0.1)),
        "max_bps": float(cfg.get("costs", {}).get("max_bps", 50.0)),
    }
    # --- parâmetros de risco usados no kill e na reentrada ---
    risk_cfg = cfg.get("risk", {})
    mdd_lookback = int(risk_cfg.get("mdd_lookback", 90))
    mdd_thres    = float(risk_cfg.get("mdd_thres", -0.20))
    vol_mult     = float(risk_cfg.get("vol_mult", 1.8))
    cooldown_days = int(risk_cfg.get("cooldown_days", 21))
    reentry_hysteresis = float(risk_cfg.get("reentry_hysteresis", 0.05))  # 5pp

    universe_cfg = cfg.get("universe", {})
    logger.info("Selecting tradable universe with hysteresis")
    universe_map = select_universe(
        panel,
        top_n=int(universe_cfg.get("top_n", 20)),
        adv_min=float(universe_cfg.get("adv_min", 0)),
        price_min=float(universe_cfg.get("price_min", 0)),
        age_min=int(universe_cfg.get("age_min", 20)),
        hysteresis_rebalances=int(universe_cfg.get("hysteresis_rebalances", 2)),
        calendar=universe_cfg.get("calendar", "BM"),
    )
    rebalance_dates = sorted(universe_map.keys())

    all_assets = list(prices.columns)
    state = _initial_state(prices.index, all_assets)
    kill_triggered = False
    cooldown = 0  # evita UnboundLocalError e controla a reentrada

    # --- Acumuladores de capacidade ---
    cap_days: int = 0
    cap_bind_days: int = 0
    turnover_cut_sum: float = 0.0
    # ----------------------------------

    cov_dates = sorted(cov_dict.keys())

    logger.info("Starting backtest loop over %d trading days", len(prices.index))
    for date in prices.index:
        returns_today = returns.loc[date]
        weights_vector = state.current_weights.reindex(all_assets, fill_value=0.0)
        portfolio_ret = float((returns_today * weights_vector).sum())
        state.portfolio_returns.loc[date] = portfolio_ret
        state.equity *= 1.0 + portfolio_ret
        state.equity_curve.loc[date] = state.equity

        rolling_vol = state.portfolio_returns.loc[:date].tail(vol_window).std(ddof=0)
        state.vol_series.loc[date] = (
            rolling_vol * np.sqrt(252) if not np.isnan(rolling_vol) else np.nan
        )

        # --- cálculo do MDD em janela e regra de reentrada com histerese ---
        trailing = state.equity_curve.loc[:date].tail(mdd_lookback).dropna()
        if len(trailing) >= 2:
            dd = trailing / trailing.cummax() - 1.0
            rolling_mdd = float(dd.min())
        else:
            rolling_mdd = 0.0
        curr_vol = float(state.vol_series.loc[date]) if not np.isnan(state.vol_series.loc[date]) else np.nan

        if kill_triggered:
            cooldown = max(0, cooldown - 1)
            ok_mdd = (rolling_mdd > (mdd_thres + reentry_hysteresis))
            ok_vol = (np.isnan(curr_vol)) or (curr_vol <= vol_mult * target_vol)
            if cooldown == 0 and ok_mdd and ok_vol:
                logger.info("Kill switch lifted on %s (mdd=%.2f, vol_ok=%s)",
                            date.date(), rolling_mdd, str(ok_vol))
                kill_triggered = False

        # if kill_triggered:
        #     state.weights_history[date] = state.current_weights.reindex(
        #         all_assets, fill_value=0.0
        #     )
        #     continue

        if date not in rebalance_dates:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        universe_assets = universe_map.get(date, [])
        if not universe_assets:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        logger.info("Rebalance on %s with %d assets", date.date(), len(universe_assets))

        ks_flag = risk_controls.kill_switch(
            state.equity_curve.loc[:date],
            state.vol_series.loc[:date],
            target_vol,
            mdd_lookback=mdd_lookback,
            mdd_thres=mdd_thres,
            vol_mult=vol_mult,
        )
        if ks_flag:
            logger.warning("Kill switch triggered on %s", date.date())
            kill_triggered = True
            cooldown = cooldown_days
            target_weights = pd.Series(0.0, index=all_assets)
        else:
            target_weights, cap_info = _compute_target_weights(
                date=date,
                universe=universe_assets,
                mix_df=mix_df,
                atr_df=atr_df,
                returns=returns,
                cov_dict=cov_dict,
                cov_dates=cov_dates,
                prices=prices,
                alpha_weights=state.current_weights,
                cfg=cfg,
                target_vol=target_vol,
            )

        if target_weights is None:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

        # --- métricas de binding de participation cap (por-ativo/cluster) ---
        try:
            cap_days += 1
            if cap_info.get("cap_bind", False):
                cap_bind_days += 1
        except Exception:
            pass
        # ---------------------------------------------------------------------
        # --- turnover cap: medir corte fracionário ---
        aligned_prev = state.current_weights.reindex(target_weights.index).fillna(0.0)
        raw_turnover = 0.5 * (target_weights - aligned_prev).abs().sum()
        target_weights = risk_controls.apply_turnover_cap(
            state.current_weights.reindex(target_weights.index, fill_value=0.0),
            target_weights,
            cap=turnover_cap,
        )
        post_turnover = 0.5 * (target_weights - aligned_prev).abs().sum()
        if raw_turnover > 1e-12:
            cut_frac = max(0.0, 1.0 - float(post_turnover / raw_turnover))
            turnover_cut_sum += cut_frac
        # ---------------------------------------------
        target_weights = target_weights.reindex(all_assets, fill_value=0.0)

        trades = _execute_portfolio_trade(
            date=date,
            prev_weights=state.current_weights,
            target_weights=target_weights,
            equity=state.equity,
            prices=prices,
            adv_notional=adv_notional,
            fee_bps=fee_bps,
            slip_params=slip_params,
        )
        state.trades.extend(trades)
        trade_costs = sum(
            (trade.get("fees", 0.0) or 0.0) + (trade.get("slip", 0.0) or 0.0)
            for trade in trades
        )
        state.equity -= trade_costs
        state.current_weights = target_weights
        state.weights_history[date] = state.current_weights.reindex(
            all_assets, fill_value=0.0
        )
        logger.info(
            "Post-trade equity %.4f; net turnover %.2f%%",
            state.equity,
            sum(abs(trade["fill_qty"]) for trade in trades),
        )

    weight_df = (
        pd.DataFrame(state.weights_history).T.reindex(prices.index).ffill().fillna(0.0)
    )
    equity_curve = state.equity_curve
    trades_df = pd.DataFrame(state.trades)

    kpis = _compute_kpis(state.portfolio_returns, equity_curve)

    # --- resumo de capacidade para consumers (capacity.py, etc.) ---
    cap_summary = {
        "cap_days": int(cap_days),
        "cap_bind_days": int(cap_bind_days),
        "cap_bind_rate": (cap_bind_days / cap_days) if cap_days else 0.0,
        "avg_turnover_cut_frac": (turnover_cut_sum / cap_days) if cap_days else 0.0,
    }
    # ----------------------------------------------------------------

    return {
        "equity_curve": equity_curve,
        "daily_positions": weight_df,
        "trades": trades_df,
        "weights": state.current_weights,
        "kpis": kpis,
        "meta": {"tda_params": tfi_meta, "tfi_stats": tfi_stats, "capacity": cap_summary},
    }


def _initial_state(index: pd.Index, assets: Iterable[str]) -> BacktestState:
    zero_weights = pd.Series(0.0, index=list(assets), dtype=float)
    return BacktestState(
        equity=1.0,
        current_weights=zero_weights,
        portfolio_returns=pd.Series(index=index, dtype=float),
        equity_curve=pd.Series(index=index, dtype=float),
        vol_series=pd.Series(index=index, dtype=float),
        weights_history={},
        trades=[],
    )


def _compute_target_weights(
    date: pd.Timestamp,
    universe: List[str],
    mix_df: pd.DataFrame,
    atr_df: pd.DataFrame,
    returns: pd.DataFrame,
    cov_dict: Dict[pd.Timestamp, pd.DataFrame],
    cov_dates: List[pd.Timestamp],
    prices: pd.DataFrame,
    alpha_weights: pd.Series,
    cfg: Dict,
    target_vol: float,
) -> Tuple[Optional[pd.Series], Dict[str, float]]:
    cap_info: Dict[str, float] = {}
    # --- caps: sanitize e defaults robustos ---
    max_asset = float(cfg.get("participation_cap", 0.10))
    risk_cfg = cfg.get("risk", {}) or {}
    raw_cluster = risk_cfg.get("max_cluster", None)
    if raw_cluster is None:
        # default: 3x o cap por ativo
        max_cluster = max(3.0 * max_asset, 1e-6)
    else:
        max_cluster = float(raw_cluster)
    # se vier <=0 por erro de config/merge, repara e loga
    if max_asset <= 0:
        logger.warning("participation_cap <= 0; usando fallback 0.10")
        max_asset = 0.10
    if max_cluster <= 0:
        logger.warning("risk.max_cluster <= 0; usando 3x participation_cap")
        max_cluster = max(3.0 * max_asset, 1e-6)
    clusters = cfg.get("clusters")

    cov = _latest_covariance(date, cov_dict, cov_dates)
    if cov is None:
        logger.warning(
            "Skipping rebalance on %s due to missing covariance", date.date()
        )
        return None, {}

    cov = (
        cov.reindex(index=universe, columns=universe)
        .dropna(axis=0, how="any")
        .dropna(axis=1, how="any")
    )
    if cov.shape[0] < 2:
        logger.warning("Insufficient covariance coverage on %s", date.date())
        return None, {}

    graph = _build_mst_from_cov(cov)
    order = topo_seriation_from_graph(cov, graph)
    hrp_weights = hrp_weights_from_order(cov, order)
    # --- ABLATION: ignorar HRP (peso = 1/N) ---
    if os.getenv("ABLATE_NO_HRP", "0") == "1":
        hrp_weights = pd.Series(1.0 / len(hrp_weights), index=hrp_weights.index)

    # mix_row = mix_df.loc[:date].tail(1)
    # if mix_row.empty:
    #     mix_adjusted = pd.Series(1.0, index=hrp_weights.index)
    # else:
    #     mix_row = mix_row.iloc[0].reindex(hrp_weights.index).fillna(0.0)
    #     mix_adjusted = (1.0 + mix_row).clip(lower=0.0)
    #     if mix_adjusted.sum() == 0:
    #         mix_adjusted = pd.Series(1.0, index=hrp_weights.index)

    # blended = hrp_weights * mix_adjusted
    # if blended.sum() == 0:
    #     blended = hrp_weights
    # blended /= blended.sum()
    mix_row = mix_df.loc[:date].tail(1)
    if mix_row.empty:
        mix_adj = pd.Series(1.0, index=hrp_weights.index)
    else:
        # s = mix_row.iloc[0].reindex(hrp_weights.index).fillna(0.0)
        # # z-score cross-section (só para shape; já é winsorizado em mix_scores)
        # std = float(s.std(ddof=0)) or 1.0
        # z = (s - float(s.mean())) / std
        # # softmax com temperatura (quanto menor T, maior contraste)
        # T = float(cfg.get("factors", {}).get("softmax_T", 1.0))
        # exps = np.exp(z / max(T, 1e-6))
        # mix_adj = pd.Series(exps / exps.sum(), index=hrp_weights.index)
        
        s = mix_row.iloc[0].reindex(hrp_weights.index).fillna(0.0)
        # Permitimos override por variável de ambiente
        T_env = os.getenv("SOFTMAX_T")
        T = float(T_env) if T_env is not None else float(cfg.get("factors", {}).get("softmax_T", 0.7))
        exps = np.exp(s / max(T, 1e-6))
        mix_adj = pd.Series(exps / exps.sum(), index=hrp_weights.index)

    # Combina HRP com forma do mix (produto seguido de renormalização)
    blended = hrp_weights * mix_adj
    blended = blended / blended.sum() if blended.sum() != 0 else hrp_weights

    #capped = risk_controls.apply_caps(
    # --- detectar se o participation cap/cluster cap irá “bater” ---
    asset_bind = bool((np.abs(blended) > (max_asset + 1e-12)).any())
    cluster_bind = False
    if clusters:
        for assets in clusters.values():
            assets = [a for a in assets if a in blended.index]
            if assets:
                if abs(float(blended.loc[assets].sum())) > (max_cluster + 1e-12):
                    cluster_bind = True
                    break
    capped = risk_controls.apply_caps(
        blended,
        max_asset=max_asset,
        max_cluster=max_cluster,
        clusters=clusters,
    )

    # atr_slice = atr_df.loc[:date]
    # if atr_slice.empty:
    #     logger.warning("ATR unavailable on %s", date.date())
    #     return None

    # risk_norm = atr_risk_normalize(capped, atr_slice.tail(1))
    # scaled = scale_to_vol(risk_norm, returns.loc[:date], target_vol=target_vol)
    # --- ABLATION: pular normalização por ATR ---
    if os.getenv("ABLATE_NO_ATR", "0") == "1":
        base_weights = capped
    else:
        atr_slice = atr_df.loc[:date]
        if atr_slice.empty:
            logger.warning("ATR unavailable on %s", date.date())
            return None, {}
        base_weights = atr_risk_normalize(capped, atr_slice.tail(1))
    # alvo de vol segue ativo (escala uniforme)
    #scaled = scale_to_vol(base_weights, returns.loc[:date], target_vol=target_vol)
    # target_vol é anual no YAML; scale_to_vol espera diária (cov diária)
    scaled = scale_to_vol(base_weights, returns.loc[:date], target_vol=target_vol  )##/ np.sqrt(252.0))
    cap_info["cap_bind"] = bool(asset_bind or cluster_bind)
    # devolvemos também o dicionário com o flag de binding
    return scaled, cap_info


def _latest_covariance(
    date: pd.Timestamp,
    cov_dict: Dict[pd.Timestamp, pd.DataFrame],
    cov_dates: List[pd.Timestamp],
) -> Optional[pd.DataFrame]:
    eligible = [dt for dt in cov_dates if dt <= date]
    if not eligible:
        return None
    return cov_dict[eligible[-1]]


def _build_mst_from_cov(cov: pd.DataFrame) -> nx.Graph:
    corr = cov.corr().fillna(0.0)
    distance = np.sqrt(0.5 * (1.0 - corr.clip(-1.0, 1.0)))
    graph = nx.Graph()
    for asset in corr.columns:
        graph.add_node(asset)
    for i, asset_i in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            asset_j = corr.columns[j]
            weight = float(distance.iloc[i, j])
            graph.add_edge(asset_i, asset_j, weight=weight)
    return nx.minimum_spanning_tree(graph, weight="weight")


def _execute_portfolio_trade(
    date: pd.Timestamp,
    prev_weights: pd.Series,
    target_weights: pd.Series,
    equity: float,
    prices: pd.DataFrame,
    adv_notional: pd.DataFrame,
    fee_bps: float,
    slip_params: Dict[str, float],
) -> List[Dict[str, float]]:
    trades: List[Dict[str, float]] = []
    price_row = prices.loc[date]
    adv_row = (
        adv_notional.loc[date]
        if date in adv_notional.index
        else pd.Series(index=price_row.index, data=np.nan)
    )

    for asset in target_weights.index:
        prev_w = float(prev_weights.get(asset, 0.0))
        new_w = float(target_weights.get(asset, 0.0))
        if abs(prev_w - new_w) < 1e-9:
            continue

        price = float(price_row.get(asset, np.nan))
        if np.isnan(price) or price == 0:
            continue
        adv_val = float(adv_row.get(asset, np.nan))
        if np.isnan(adv_val) or adv_val <= 0:
            adv_val = abs(new_w * equity) * 10.0

        prev_qty = prev_w * equity / price
        target_qty = new_w * equity / price

        execution = execute_trade(
            prev_qty, target_qty, price, adv_val, fee_bps, slip_params
        )
        execution.update(
            {
                "asset": asset,
                "date": date,
                "price": price,
            }
        )
        trades.append(execution)

    return trades


def _compute_kpis(
    portfolio_returns: pd.Series, equity_curve: pd.Series
) -> Dict[str, float]:
    returns = portfolio_returns.fillna(0.0)
    equity = equity_curve.ffill().dropna()
    if equity.empty:
        return {"final_equity": 1.0}

    ending_equity = float(equity.iloc[-1])
    total_return = ending_equity - 1.0
    periods = len(returns)
    if periods and ending_equity > 0:
        ann_return = ending_equity ** (252 / periods) - 1.0
    else:
        ann_return = np.nan
    ann_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = (
        ann_return / ann_vol if ann_vol > 0 and not np.isnan(ann_return) else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return {
        "final_equity": ending_equity,
        "total_return": float(total_return),
        "annual_return": float(ann_return) if not np.isnan(ann_return) else np.nan,
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": max_dd,
    }
