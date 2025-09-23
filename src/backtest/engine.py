from __future__ import annotations

"""Core backtesting engine scaffolding."""

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from backtest.execution import execute_trade
from dataio.loaders import get_adv, get_panel, select_universe
from features import TFIParams, mix_scores, momentum_12_1, quality_proxy, tfi_score
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

    logger.info("Pre-computing factor scores")
    tfi_params = cfg.get("tda", {})
    tfi_cfg = TFIParams(
        delay=int(tfi_params.get("delay", 1)),
        dim=int(tfi_params.get("dim", 3)),
        n_cubes=int(tfi_params.get("n_cubes", 4)),
        overlap=float(tfi_params.get("overlap", 0.5)),
        epsilon=float(tfi_params.get("epsilon", 0.5)),
        min_samples=int(tfi_params.get("min_samples", 3)),
        window=int(tfi_params.get("window", max(vol_window, 63))),
    )

    regime_series = (
        tfi_score(prices, params=tfi_cfg).reindex(prices.index).ffill().fillna(0.0)
    )
    momentum_df = momentum_12_1(prices).reindex(prices.index).ffill().fillna(0.0)
    quality_df = quality_proxy(prices).reindex(prices.index).ffill().fillna(0.0)

    factors_cfg = cfg.get("factors", {})
    alphas = factors_cfg.get("alphas", [0.6, 0.3, 0.1])
    if alphas and isinstance(alphas[0], (int, float)):
        alpha, beta, gamma = (list(alphas) + [0.3, 0.1])[:3]
    else:
        alpha, beta, gamma = 0.6, 0.3, 0.1

    common_index = regime_series.index.intersection(momentum_df.index).intersection(
        quality_df.index
    )
    if common_index.empty:
        mix_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    else:
        regime_series = regime_series.reindex(common_index)
        momentum_df = momentum_df.reindex(common_index)
        quality_df = quality_df.reindex(common_index)
        mix_df = mix_scores(regime_series, momentum_df, quality_df, alpha, beta, gamma)
        mix_df = mix_df.rename(columns=lambda c: c.replace("mix_", ""))
        mix_df = mix_df.reindex(prices.index).ffill().fillna(0.0)

    turnover_cap = float(cfg.get("turnover_cap", 0.25))
    target_vol = float(cfg.get("vol_target", 0.10))
    fee_bps = float(cfg.get("costs", {}).get("fee_bps", 5.0))
    slip_params = {
        "k": float(cfg.get("costs", {}).get("k", 0.1)),
        "max_bps": float(cfg.get("costs", {}).get("max_bps", 50.0)),
    }

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

        if kill_triggered:
            state.weights_history[date] = state.current_weights.reindex(
                all_assets, fill_value=0.0
            )
            continue

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
            mdd_lookback=int(cfg.get("risk", {}).get("mdd_lookback", 90)),
            mdd_thres=float(cfg.get("risk", {}).get("mdd_thres", -0.20)),
            vol_mult=float(cfg.get("risk", {}).get("vol_mult", 1.8)),
        )
        if ks_flag:
            logger.warning("Kill switch triggered on %s", date.date())
            kill_triggered = True
            target_weights = pd.Series(0.0, index=all_assets)
        else:
            target_weights = _compute_target_weights(
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

        target_weights = risk_controls.apply_turnover_cap(
            state.current_weights.reindex(target_weights.index, fill_value=0.0),
            target_weights,
            cap=turnover_cap,
        )
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
        state.equity += sum(trade["cash_delta"] for trade in trades)
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

    return {
        "equity_curve": equity_curve,
        "daily_positions": weight_df,
        "trades": trades_df,
        "weights": state.current_weights,
        "kpis": kpis,
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
) -> Optional[pd.Series]:
    max_asset = float(cfg.get("participation_cap", 0.10))
    max_cluster = float(cfg.get("risk", {}).get("max_cluster", 0.35))
    clusters = cfg.get("clusters")

    cov = _latest_covariance(date, cov_dict, cov_dates)
    if cov is None:
        logger.warning(
            "Skipping rebalance on %s due to missing covariance", date.date()
        )
        return None

    cov = (
        cov.reindex(index=universe, columns=universe)
        .dropna(axis=0, how="any")
        .dropna(axis=1, how="any")
    )
    if cov.shape[0] < 2:
        logger.warning("Insufficient covariance coverage on %s", date.date())
        return None

    graph = _build_mst_from_cov(cov)
    order = topo_seriation_from_graph(cov, graph)
    hrp_weights = hrp_weights_from_order(cov, order)

    mix_row = mix_df.loc[:date].tail(1)
    if mix_row.empty:
        mix_adjusted = pd.Series(1.0, index=hrp_weights.index)
    else:
        mix_row = mix_row.iloc[0].reindex(hrp_weights.index).fillna(0.0)
        mix_adjusted = (1.0 + mix_row).clip(lower=0.0)
        if mix_adjusted.sum() == 0:
            mix_adjusted = pd.Series(1.0, index=hrp_weights.index)

    blended = hrp_weights * mix_adjusted
    if blended.sum() == 0:
        blended = hrp_weights
    blended /= blended.sum()

    capped = risk_controls.apply_caps(
        blended,
        max_asset=max_asset,
        max_cluster=max_cluster,
        clusters=clusters,
    )

    atr_slice = atr_df.loc[:date]
    if atr_slice.empty:
        logger.warning("ATR unavailable on %s", date.date())
        return None

    risk_norm = atr_risk_normalize(capped, atr_slice.tail(1))
    scaled = scale_to_vol(risk_norm, returns.loc[:date], target_vol=target_vol)
    return scaled


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
