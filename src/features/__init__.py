"""Feature engineering utilities for Atlas."""

from .factors import (
    mix_scores,
    momentum_12_1,
    quality_proxy,
    slope_nd,
    get_alphas_from_cfg,
    forward_returns,
    peripherality_factor,
)
from .regime import compute_ph_regime_index
from .tda.mapper import RegimeAwareMapper

try:
    from .tda.ph_turbulence import PHTurbulenceTransformer
except ImportError:  # pragma: no cover - optional dependency (giotto-tda)
    PHTurbulenceTransformer = None

__all__ = [
    "get_alphas_from_cfg",
    "momentum_12_1",
    "slope_nd",
    "quality_proxy",
    "mix_scores",
    "forward_returns",
    "peripherality_factor",
    "compute_ph_regime_index",
    "RegimeAwareMapper",
]

if PHTurbulenceTransformer is not None:
    __all__.append("PHTurbulenceTransformer")
