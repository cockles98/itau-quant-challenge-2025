"""Feature engineering utilities for t_hrp_v3."""

from .factors import (
    mix_scores,
    momentum_12_1,
    quality_proxy,
    slope_nd,
    get_alphas_from_cfg,
    forward_returns,
    peripherality_factor,
)
from .tda import TFIParams, mapper_graph, mapper_for_asset, takens_embedding, tfi_score
from .tda.mapper import RegimeAwareMapper
from .regime import compute_ph_regime_index

try:
    from .tda.ph_turbulence import PHTurbulenceTransformer
except ImportError:  # pragma: no cover - optional dependency (giotto-tda)
    PHTurbulenceTransformer = None

__all__ = [
    "TFIParams",
    "takens_embedding",
    "mapper_graph",
    "mapper_for_asset",
    "get_alphas_from_cfg",
    "tfi_score",
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
