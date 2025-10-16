#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataio.config import load_config
from backtest.engine import run_backtest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sensitivity grid over Mapper hyperparameters (n_cubes × overlap)."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Configuration YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--n-cubes",
        nargs="*",
        type=int,
        default=[4, 6, 8, 10, 12],
        help="Grid values for number of cubes (default: %(default)s)",
    )
    parser.add_argument(
        "--overlaps",
        nargs="*",
        type=float,
        default=[0.2, 0.35, 0.5, 0.65, 0.8],
        help="Grid values for cover overlap (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="reports/tda_sensitivity",
        help="Output stem (without extension). Files saved as <stem>.csv/png/json",
    )
    parser.add_argument(
        "--reuse-equity",
        action="store_true",
        help="If set, reuse existing cached results when available (based on summary JSON).",
    )
    return parser.parse_args()


def _calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown >= 0:
        return float("nan")
    denom = abs(max_drawdown)
    return annual_return / denom if denom > 1e-9 else float("nan")


def _run_single(cfg: dict, n_cubes: int, overlap: float) -> dict:
    cfg_local = json.loads(json.dumps(cfg))  # deep copy
    mapper_cfg = cfg_local.setdefault("tda_mapper", {})
    mapper_cfg["n_cubes"] = int(n_cubes)
    mapper_cfg["overlap"] = float(overlap)

    result = run_backtest(cfg_local)
    kpis = result["kpis"]
    ann_return = float(kpis.get("annual_return", float("nan")))
    sharp = float(kpis.get("sharpe", float("nan")))
    max_dd = float(kpis.get("max_drawdown", float("nan")))
    calmar = _calmar_ratio(ann_return, max_dd)
    meta = result.get("meta", {})
    mapper_stats = meta.get("tda_topology", {}).get("regime_stats", {})

    return {
        "n_cubes": n_cubes,
        "overlap": overlap,
        "annual_return": ann_return,
        "sharpe": sharp,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "final_equity": float(kpis.get("final_equity", float("nan"))),
        "regime_mean": mapper_stats.get("mean"),
        "regime_std": mapper_stats.get("std"),
    }


def _ensure_output_paths(stem: str) -> Tuple[Path, Path, Path]:
    stem_path = Path(stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    return (
        stem_path.with_suffix(".csv"),
        stem_path.with_suffix(".png"),
        stem_path.with_suffix(".json"),
    )


def _plot_heatmaps(df: pd.DataFrame, png_path: Path) -> None:
    pivot_sharpe = df.pivot(index="n_cubes", columns="overlap", values="sharpe")
    pivot_calmar = df.pivot(index="n_cubes", columns="overlap", values="calmar")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(
        axes,
        (pivot_sharpe, pivot_calmar),
        ("Sharpe", "Calmar"),
        strict=True,
    ):
        im = ax.imshow(data, cmap="viridis", origin="lower")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([f"{c:.2f}" for c in data.columns])
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index)
        ax.set_xlabel("Overlap")
        ax.set_ylabel("n_cubes")
        ax.set_title(title)
        for (i, j), value in np.ndenumerate(data.values):
            if np.isnan(value):
                label = "nan"
            else:
                label = f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def _maybe_load_cached(csv_path: Path, reuse: bool) -> pd.DataFrame | None:
    if reuse and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if {"n_cubes", "overlap", "sharpe"}.issubset(df.columns):
                print(f"[INFO] Reusing cached sensitivity results from {csv_path}")
                return df
        except Exception:
            pass
    return None


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    csv_path, png_path, json_path = _ensure_output_paths(args.output)

    cached = _maybe_load_cached(csv_path, args.reuse_equity)
    if cached is not None:
        df = cached
    else:
        results = []
        for n_cubes in args.n_cubes:
            for overlap in args.overlaps:
                print(f"[INFO] Running grid point n_cubes={n_cubes}, overlap={overlap:.2f}")
                res = _run_single(cfg, n_cubes, overlap)
                results.append(res)
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved grid results to {csv_path}")

    _plot_heatmaps(df, png_path)
    print(f"[INFO] Saved heatmap to {png_path}")

    summary = {
        "config": args.config,
        "n_cubes": args.n_cubes,
        "overlaps": args.overlaps,
        "best_sharpe": df.loc[df["sharpe"].idxmax()].to_dict(),
        "best_calmar": df.loc[df["calmar"].idxmax()].to_dict(),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote summary to {json_path}")


if __name__ == "__main__":
    main()

