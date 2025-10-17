from __future__ import annotations

"""Build a consolidated PDF report for Atlas."""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from backtest.engine import run_backtest
from dataio.config import load_config
from dataio.loaders import get_panel
from features import compute_ph_regime_index
try:  # pragma: no cover - optional dependency managed at runtime
    from features.tda.ph_turbulence import PHTurbulenceTransformer
except ImportError:  # pragma: no cover
    PHTurbulenceTransformer = None  # type: ignore
from metrics import cagr, calmar, hit_rate, mdd, sharpe, sortino, vol
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


def _compute_regime_series(cfg: Dict, panel: pd.DataFrame) -> pd.Series:
    prices = panel["close"].unstack("asset").sort_index()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return compute_ph_regime_index(returns, cfg)


def _extract_regime_dataframe(backtest_result: Dict[str, Any]) -> Optional[pd.DataFrame]:
    meta = backtest_result.get("meta", {})
    regime_meta = meta.get("regime_controls", {})
    per_date = regime_meta.get("per_date")
    if not isinstance(per_date, dict) or not per_date:
        return None
    df = pd.DataFrame.from_dict(per_date, orient="index")
    try:
        df.index = pd.to_datetime(df.index)
    except (TypeError, ValueError):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    numeric_cols = df.select_dtypes(include="object").columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
    return df


def _load_mapper_metrics(meta: Dict[str, Any]) -> Optional[pd.DataFrame]:
    tda_meta = meta.get("tda_params") or meta.get("tda_topology") or {}
    path_str = tda_meta.get("mapper_metrics_path")
    if not path_str:
        return None
    candidates = [Path(path_str)]
    candidate_path = Path(path_str)
    if not candidate_path.is_absolute():
        candidates.append(REPORT_DIR / candidate_path.name)
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - IO failure is non-critical
            logging.warning("Failed to read mapper metrics from %s: %s", path, exc)
            continue
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date")
            df = df.drop_duplicates(subset="date", keep="last")
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index()
        return df
    return None


def _compute_turbulence_series(
    returns_df: pd.DataFrame,
    tda_cfg: Mapping[str, Any],
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    if PHTurbulenceTransformer is None:
        logging.warning(
            "PHTurbulenceTransformer unavailable; skipping turbulence computation."
        )
        return None, None
    if returns_df.empty:
        return None, None
    try:
        transformer = PHTurbulenceTransformer(
            window=int(tda_cfg.get("window", 50)),
            homology_dimensions=(int(tda_cfg.get("homology_dim", 1)),),
            norm=str(tda_cfg.get("norm", "l2")),
            compute_entropy=False,
            compute_amplitude=False,
            compute_wasserstein=False,
        )
        transformer.fit(returns_df)
        base_series = transformer.transform(returns_df)
    except Exception as exc:  # pragma: no cover - diagnostic path
        logging.warning("Failed to compute PH turbulence series: %s", exc)
        return None, None

    span = max(int(tda_cfg.get("smooth_span", 10)), 1)
    lookback = max(int(tda_cfg.get("zscore_lookback", 250)), 1)
    smoothed = base_series.ewm(span=span, adjust=False).mean()
    rolling_mean = smoothed.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = smoothed.rolling(window=lookback, min_periods=lookback).std(ddof=0)
    rolling_std = rolling_std.replace(0.0, np.nan)
    z_values = (smoothed - rolling_mean) / rolling_std
    z_values = z_values.replace([np.inf, -np.inf], np.nan)

    base_series.name = "ph_turbulence"
    z_series = z_values.rename("PH_turbulence_z")
    return base_series, z_series


def _compute_regime_kpis(
    returns: pd.Series,
    z_scores: Optional[pd.Series],
    output_path: Path,
) -> Optional[Path]:
    if z_scores is None:
        return None
    combined = pd.DataFrame({"returns": returns}).join(
        z_scores.rename("z"), how="inner"
    )
    combined = combined.dropna()
    if combined.empty:
        return None
    try:
        combined["tercile"] = pd.qcut(
            combined["z"], q=3, labels=["Low", "Mid", "High"]
        )
    except ValueError:
        combined["tercile"] = pd.cut(
            combined["z"],
            bins=3,
            labels=["Low", "Mid", "High"],
        )
    rows: List[Dict[str, Any]] = []
    for label, group in combined.groupby("tercile"):
        group_returns = group["returns"].sort_index()
        if group_returns.empty:
            continue
        equity = (1.0 + group_returns).cumprod()
        row = {
            "tercile": str(label),
            "count": int(group_returns.size),
            "avg_daily_return": float(group_returns.mean()),
            "annual_return": float(group_returns.mean() * 252),
            "CAGR": cagr(equity),
            "Vol": vol(group_returns),
            "Sharpe": sharpe(group_returns),
            "Sortino": sortino(group_returns),
            "MaxDD": mdd(equity),
            "Calmar": calmar(equity),
            "HitRate": hit_rate(group_returns),
            "mean_z": float(group["z"].mean()),
        }
        rows.append(row)
    if not rows:
        return None
    table = pd.DataFrame(rows).set_index("tercile").sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path)
    return output_path


def _build_tda_dashboard(
    z_series: Optional[pd.Series],
    mapper_metrics: Optional[pd.DataFrame],
    regime_df: Optional[pd.DataFrame],
    *,
    alert_sigma: float,
    riskoff_sigma: float,
    output_path: Path,
) -> Optional[Path]:
    has_turbulence = z_series is not None and not z_series.dropna().empty
    has_mapper = (
        mapper_metrics is not None
        and not mapper_metrics.empty
        and "n_components" in mapper_metrics.columns
    )
    has_regime = regime_df is not None and not regime_df.empty
    if not any([has_turbulence, has_mapper, has_regime]):
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax0 = axes[0]
    ax0.set_title("PH Turbulence Z-Score")
    if has_turbulence:
        z_clean = z_series.dropna()
        ax0.plot(z_clean.index, z_clean.values, label="PH_turbulence_z", color="C0")
        ax0.axhline(alert_sigma, color="orange", linestyle="--", label="alert σ")
        ax0.axhline(riskoff_sigma, color="red", linestyle="--", label="risk-off σ")
        ax0.legend(loc="upper left")
    else:
        ax0.text(
            0.5,
            0.5,
            "No turbulence data",
            transform=ax0.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    ax0.grid(True, linestyle="--", alpha=0.3)

    ax1 = axes[1]
    ax1.set_title("Mapper Fragmentation (n_components)")
    if has_mapper:
        mapper_series = (
            mapper_metrics["n_components"].astype(float).dropna().sort_index()
        )
        ax1.step(mapper_series.index, mapper_series.values, where="post", color="C1")
        ax1.set_ylabel("Components")
    else:
        ax1.text(
            0.5,
            0.5,
            "No Mapper metrics available",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = axes[2]
    ax2.set_title("Risk Controls: Target Vol vs Gross Exposure")
    if has_regime and "target_vol_eff" in regime_df.columns:
        regime_sorted = regime_df.sort_index()
        target_series = (
            regime_sorted["target_vol_eff"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not target_series.empty:
            ax2.plot(
                target_series.index,
                target_series.values,
                label="target_vol_eff",
                color="C2",
            )
        gross_series = (
            regime_sorted.get("gross_target", pd.Series(dtype=float))
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not gross_series.empty:
            ax2.plot(
                gross_series.index,
                gross_series.values,
                label="gross_target",
                color="C3",
            )
        if not target_series.empty and not gross_series.empty:
            ratio = gross_series.reindex(target_series.index).div(
                target_series.replace(0.0, np.nan)
            )
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                ax2.plot(
                    ratio.index,
                    ratio.values,
                    label="gross/target ratio",
                    color="C4",
                    linestyle="--",
                )
        ax2.legend(loc="upper left")
    else:
        ax2.text(
            0.5,
            0.5,
            "No regime telemetry available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_pdf(config_path: Path, output_path: Path) -> None:
    cfg = load_config(str(config_path))
    panel = _load_panel(cfg)
    prices = panel["close"].unstack("asset").sort_index()
    asset_returns = (
        prices.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    logging.info("Running baseline backtest...")
    backtest_result = run_backtest(cfg, panel=panel)
    equity = backtest_result["equity_curve"]
    returns = equity.pct_change().dropna()
    equity_plot = plot_equity_curves(
        {"Strategy": equity}, filename=f"equity_curve_{_timestamp_tag()}.png"
    )
    table_kpis({"Strategy": {"equity_curve": equity, "returns": returns}})
    kpi_csv_path = REPORT_DIR / "kpi_table.csv"

    meta = backtest_result.get("meta", {})
    regime_df = _extract_regime_dataframe(backtest_result)
    mapper_metrics_df = _load_mapper_metrics(meta)
    tda_ph_cfg = cfg.get("tda_ph", {})
    _, turbulence_z = _compute_turbulence_series(asset_returns, tda_ph_cfg)

    regime_kpi_path = _compute_regime_kpis(
        returns,
        turbulence_z,
        REPORT_DIR / "regime_kpis.csv",
    )
    tda_dashboard_path = _build_tda_dashboard(
        turbulence_z,
        mapper_metrics_df,
        regime_df,
        alert_sigma=float(tda_ph_cfg.get("alert_sigma", 1.0)),
        riskoff_sigma=float(tda_ph_cfg.get("riskoff_sigma", 2.0)),
        output_path=REPORT_DIR / "tda_dashboard.png",
    )

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

    logging.info("Computing regime subperiod metrics...")
    regime_series = _compute_regime_series(cfg, panel)
    regime_subperiod_path = None
    if regime_series.notna().any():
        from validation import regime_subperiods

        regime_table = regime_subperiods(cfg, panel, regime_series)
        regime_subperiod_path = REPORT_DIR / "regime_subperiods.csv"
        regime_table.to_csv(regime_subperiod_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "Atlas - o cartografo do mercado")
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

    if tda_dashboard_path:
        img_width, img_height = _image_dims(tda_dashboard_path)
        if y - img_height < margin:
            c.showPage()
            y = height - margin
        c.drawImage(
            str(tda_dashboard_path),
            margin,
            y - img_height,
            width=img_width,
            height=img_height,
        )
        y -= img_height + 20

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
            f"TDA dashboard: {tda_dashboard_path}"
            if tda_dashboard_path
            else "TDA dashboard: n/a"
        ),
        (
            f"Regime tercile KPIs: {regime_kpi_path}"
            if regime_kpi_path
            else "Regime tercile KPIs: n/a"
        ),
        (
            f"Sensitivity heatmaps: {[str(p) for p in heatmap_paths]}"
            if heatmap_paths
            else "Sensitivity heatmaps: n/a"
        ),
        f"Capacity outputs: {capacity_text}",
        f"Stress costs CSV: {stress_path}",
        (
            f"Regime subperiods: {regime_subperiod_path}"
            if regime_subperiod_path
            else "Regime subperiods: n/a"
        ),
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
