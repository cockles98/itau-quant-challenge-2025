"""Minimal exports for topological utilities."""

from __future__ import annotations

from .mapper import MapperMetrics, RegimeAwareMapper

try:  # pragma: no cover - optional dependency (giotto-tda)
    from .ph_turbulence import PHTurbulenceTransformer
except ImportError:  # pragma: no cover
    PHTurbulenceTransformer = None  # type: ignore[assignment]

__all__ = ["MapperMetrics", "RegimeAwareMapper"]

if PHTurbulenceTransformer is not None:
    __all__.append("PHTurbulenceTransformer")
