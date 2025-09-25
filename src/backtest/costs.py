from __future__ import annotations

from typing import Union

import logging

Number = Union[int, float]

__all__ = ["commission_cost", "slippage_cost"]

_LOG = logging.getLogger(__name__)


def commission_cost(notional: Number, fee_bps: Number) -> float:
    """Return the absolute commission cost for a trade notional.

    Parameters
    ----------
    notional : Number
        Monetary value of the trade (price * quantity). Sign is ignored.
    fee_bps : Number
        Commission rate in basis points (1 bps = 0.0001).
    """

    if fee_bps < 0:
        raise ValueError("fee_bps must be non-negative")

    return abs(float(notional)) * float(fee_bps) / 10_000.0


def slippage_cost(
    notional: Number,
    adv: Number,
    *,
    k: float = 0.1,
    max_bps: float = 50.0,
) -> float:
    """Compute a simple quadratic-style slippage cost based on participation.

    Slippage is proportional to the fraction of ADV crossed and capped at
    ``max_bps``. Participation above 5% of ADV emits a warning as it typically
    indicates deteriorating execution quality.
    """

    if adv <= 0:
        raise ValueError("adv must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    if max_bps < 0:
        raise ValueError("max_bps must be non-negative")

    notional_abs = abs(float(notional))
    participation = notional_abs / float(adv)

    if participation > 0.05:
        _LOG.warning(
            "Participation %.2f%% exceeds 5%% ADV threshold.",
            participation * 100,
        )

    impact_bps = min(max_bps, k * 10_000.0 * participation)
    return notional_abs * impact_bps / 10_000.0
