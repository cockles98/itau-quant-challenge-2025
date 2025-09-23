from __future__ import annotations

from typing import Dict, Mapping, Union

from .costs import commission_cost, slippage_cost

Number = Union[int, float]

__all__ = ["execute_trade"]


def execute_trade(
    prev_qty: Number,
    target_qty: Number,
    price: Number,
    adv: Number,
    fee_bps: Number,
    slip_params: Mapping[str, Number] | None = None,
) -> Dict[str, float]:
    """Execute a single-period trade and return execution statistics.

    Parameters
    ----------
    prev_qty : Number
        Previous position size.
    target_qty : Number
        Desired position size after the trade.
    price : Number
        Execution price per unit. Must be positive.
    adv : Number
        Average daily volume in the same notional units as ``price * quantity``.
    fee_bps : Number
        Commission rate in basis points.
    slip_params : Mapping[str, Number], optional
        Parameters passed to :func:`slippage_cost` (``k`` and ``max_bps``).

    Returns
    -------
    dict
        Keys: ``fill_qty`` (executed quantity), ``cash_delta`` (cash impact of
        trade), ``fees`` (commission cost), ``slip`` (slippage cost).
    """

    if price <= 0:
        raise ValueError("price must be positive")

    trade_qty = float(target_qty) - float(prev_qty)
    notional = trade_qty * float(price)

    fees = commission_cost(notional, fee_bps)

    params = dict(slip_params or {})
    slip = slippage_cost(
        notional,
        adv,
        k=float(params.get("k", 0.1)),
        max_bps=float(params.get("max_bps", 50.0)),
    )

    cash_delta = -notional - fees - slip

    return {
        "fill_qty": trade_qty,
        "cash_delta": cash_delta,
        "fees": fees,
        "slip": slip,
    }
