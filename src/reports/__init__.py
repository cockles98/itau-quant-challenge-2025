"""Reporting package exports."""

from .reporting import plot_equity_curves, table_kpis

__all__ = ["plot_equity_curves", "table_kpis", "build_pdf"]


def build_pdf(*args, **kwargs):
    from .build_pdf import build_pdf as _build_pdf

    return _build_pdf(*args, **kwargs)
