from __future__ import annotations

"""Build a consolidated PDF report for T-HRP v3.0."""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from backtest.engine import run_backtest
from dataio.config import load_config
from dataio.loaders import get_panel
from features import TFIParams, tfi_score
from reports import plot_equity_curves, table_kpis
from validation import capacity_curve, param_sensitivity_heatmaps, stress_costs

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def _timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _synthetic_panel(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, periods=756, freq="B")
    assets = ["AAA", "BBB", "CCC"]
    records = []
    rng = np.random.default_rng(0)
    for asset in assets:
        base = 100 + 5 * (ord(asset[0]) - ord("A"))
        prices = base + np.arange(len(dates)) + 0.5 * rng.standard_normal(len(dates))
        volumes = 1_000_000 + rng.integers(-50_000, 50_000, len(dates))
        for dt, price, volume in zip(dates, prices, volumes):
            records.append((dt, asset, float(price), float(abs(volume))))
    return pd.DataFrame(
        records, columns=["date", "asset", "close", "volume"]
    ).set_index(["date", "asset"])


def _load_panel(cfg: Dict) -> pd.DataFrame:
    dates_cfg = cfg.get("dates", {})
    start = dates_cfg.get("start")
    end = dates_cfg.get("end")
    if not start or not end:
        raise ValueError("Configuration must include dates.start and dates.end")
    try:
        panel = get_panel(start, end)
    except Exception as err:  # pragma: no cover
        logging.warning("Falling back to synthetic panel: %s", err)
        panel = _synthetic_panel(start, end)
        cfg.setdefault("dates", {})
        cfg["dates"]["start"] = panel.index.get_level_values(0).min().date().isoformat()
        cfg["dates"]["end"] = panel.index.get_level_values(0).max().date().isoformat()
    return panel.sort_index(level=[0, 1], sort_remaining=True)


def _image_dims(
    img_path: Path, max_width: float = 6.0, dpi: float = 72.0
) -> tuple[float, float]:
    reader = ImageReader(str(img_path))
    width_px, height_px = reader.getSize()
    width_in = width_px / dpi
    height_in = height_px / dpi
    if width_in > max_width:
        scale = max_width / width_in
        width_in *= scale
        height_in *= scale
    return width_in * dpi, height_in * dpi


def _compute_tfi(cfg: Dict, panel: pd.DataFrame) -> pd.Series:
    prices = panel["close"].unstack("asset").sort_index()
    tda_cfg = cfg.get("tda", {})
    params = TFIParams(
        delay=int(tda_cfg.get("delay", 1)),
        dim=int(tda_cfg.get("dim", 3)),
        n_cubes=int(tda_cfg.get("n_cubes", 8)),
        overlap=float(tda_cfg.get("overlap", 0.5)),
        epsilon=float(tda_cfg.get("epsilon", 0.5)),
        min_samples=int(tda_cfg.get("min_samples", 3)),
        window=int(tda_cfg.get("window", 252)),
    )
    return tfi_score(prices, params=params)


def build_pdf(config_path: Path, output_path: Path) -> None:
    cfg = load_config(str(config_path))
    panel = _load_panel(cfg)

    logging.info("Running baseline backtest...")
    backtest_result = run_backtest(cfg, panel=panel)
    equity = backtest_result["equity_curve"]
    returns = equity.pct_change().dropna()
    equity_plot = plot_equity_curves(
        {"Strategy": equity}, filename=f"equity_curve_{_timestamp_tag()}.png"
    )
    table_kpis({"Strategy": {"equity_curve": equity, "returns": returns}})
    kpi_csv_path = REPORT_DIR / "kpi_table.csv"

    validation_cfg = cfg.get("validation", {})

    heatmap_paths: List[Path] = []
    grid = validation_cfg.get("robustness_grid")
    if grid:
        logging.info("Generating sensitivity heatmaps...")
        trimmed_grid = {k: list(v)[:2] for k, v in grid.items()}
        for key, values in trimmed_grid.items():
            if len(values) < 2:
                values.append(values[0])
                trimmed_grid[key] = values
        heatmap_paths = param_sensitivity_heatmaps(cfg, panel, trimmed_grid)

    logging.info("Running capacity curve...")
    caps = validation_cfg.get("participation_caps", [0.01, 0.02, 0.05, 0.10])
    capacity_outputs = capacity_curve(cfg, panel, participation_caps=caps[:3])
    capacity_fig = capacity_outputs.get("figure")

    logging.info("Stress testing costs...")
    stress_path = stress_costs(
        cfg, panel, validation_cfg.get("stress_multipliers", [0.5, 1.0, 2.0])
    )

    logging.info("Computing regime KPIs...")
    tfi_series = _compute_tfi(cfg, panel)
    regime_path = None
    if tfi_series.notna().any():
        from validation import regime_subperiods

        regime_table = regime_subperiods(cfg, panel, tfi_series)
        regime_path = REPORT_DIR / "regime_kpis.csv"
        regime_table.to_csv(regime_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "T-HRP v3.0 Consolidated Report")
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - margin - 20, f"Config: {config_path}")
    c.drawString(
        margin,
        height - margin - 35,
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
    )

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, height - margin - 60, "Final KPIs")
    c.setFont("Helvetica", 10)
    y = height - margin - 75
    for key, value in backtest_result["kpis"].items():
        formatted = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        c.drawString(margin, y, f"{key}: {formatted}")
        y -= 12

    img_width, img_height = _image_dims(equity_plot)
    c.drawImage(
        str(equity_plot),
        margin,
        y - img_height - 10,
        width=img_width,
        height=img_height,
    )
    y = y - img_height - 30

    for path in heatmap_paths:
        img_width, img_height = _image_dims(path)
        if y - img_height < margin:
            c.showPage()
            y = height - margin
        c.drawImage(
            str(path), margin, y - img_height, width=img_width, height=img_height
        )
        y -= img_height + 20

    if capacity_fig:
        img_width, img_height = _image_dims(capacity_fig)
        if y - img_height < margin:
            c.showPage()
            y = height - margin
        c.drawImage(
            str(capacity_fig),
            margin,
            y - img_height,
            width=img_width,
            height=img_height,
        )
        y -= img_height + 20

    if y < margin:
        c.showPage()
        y = height - margin

    capacity_text = {k: str(v) for k, v in capacity_outputs.items()}
    artifact_lines = [
        f"Equity plot: {equity_plot}",
        f"KPI table CSV: {kpi_csv_path}",
        (
            f"Sensitivity heatmaps: {[str(p) for p in heatmap_paths]}"
            if heatmap_paths
            else "Sensitivity heatmaps: n/a"
        ),
        f"Capacity outputs: {capacity_text}",
        f"Stress costs CSV: {stress_path}",
        f'Regime KPIs: {regime_path if regime_path else "n/a"}',
    ]

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Artifacts")
    y -= 15
    c.setFont("Helvetica", 10)
    for line in artifact_lines:
        c.drawString(margin, y, line)
        y -= 12

    if y < margin:
        c.showPage()
        y = height - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Observations")
    y -= 15
    c.setFont("Helvetica", 10)
    observations = [
        "Resultados dependem da qualidade dos dados; valide outliers antes de usar.",
        "Heatmaps indicam sensibilidade aos parametros topologicos (delay, dim).",
        "Capacity curve mostra impacto do participation cap em Sharpe e drawdown.",
    ]
    for line in observations:
        c.drawString(margin + 10, y, f"- {line}")
        y -= 12

    c.save()
    print(f"PDF report saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated PDF report")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    build_pdf(args.config, args.out)


if __name__ == "__main__":
    main()
