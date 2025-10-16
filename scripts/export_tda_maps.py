#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from dataio.config import load_config
from dataio.loaders import get_panel
from features import RegimeAwareMapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Mapper graphs (PNG + JSON) for a sequence of rebalance dates."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Configuration YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date (YYYY-MM-DD) overriding config.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date (YYYY-MM-DD) overriding config.",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Specific dates to export (space separated). Overrides --frequency if supplied.",
    )
    parser.add_argument(
        "--frequency",
        default="M",
        help="Pandas offset alias for sampling dates (default: %(default)s). Examples: W, M, BM.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback window in trading days. Defaults to mapper config or 252.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/tda/maps",
        help="Directory for artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--lens",
        choices=("pca_umap", "volatility", "beta_selic", "beta_usd", "beta_ipca", "custom"),
        help="Override mapper lens.",
    )
    parser.add_argument("--n-cubes", type=int, help="Override number of cubes.")
    parser.add_argument("--overlap", type=float, help="Override cover overlap.")
    parser.add_argument("--min-cluster-size", type=int, help="Override DBSCAN min_samples.")
    parser.add_argument("--eps-quantile", type=float, help="Override DBSCAN eps quantile.")
    parser.add_argument(
        "--random-state",
        type=int,
        help="Random state override for reproducible layouts.",
    )
    parser.add_argument(
        "--layout",
        choices=("spring", "kamada"),
        default="spring",
        help="Graph layout algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output figure DPI (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra logging information.",
    )
    return parser.parse_args()


def _resolve_dates(
    prices: pd.DataFrame,
    explicit_dates: Iterable[str] | None,
    frequency: str,
) -> list[pd.Timestamp]:
    if explicit_dates:
        return sorted(pd.to_datetime(list(explicit_dates)))
    sampled = prices.resample(frequency).last().index
    return list(sampled)


def _prepare_mapper(args: argparse.Namespace, mapper_cfg: dict) -> tuple[RegimeAwareMapper, dict]:
    params = mapper_cfg.copy()
    overrides = {
        "lens": args.lens,
        "n_cubes": args.n_cubes,
        "overlap": args.overlap,
        "min_cluster_size": args.min_cluster_size,
        "eps_quantile": args.eps_quantile,
        "random_state": args.random_state,
    }
    params.update({k: v for k, v in overrides.items() if v is not None})
    mapper = RegimeAwareMapper(
        n_cubes=int(params.get("n_cubes", 8)),
        overlap=float(params.get("overlap", 0.4)),
        lens=str(params.get("lens", "pca_umap")),
        min_cluster_size=int(params.get("min_cluster_size", 3)),
        eps_quantile=float(params.get("eps_quantile", 0.25)),
        epsilon_adaptive=bool(params.get("epsilon_adaptive", True)),
        random_state=params.get("random_state"),
    )
    return mapper, params


def _layout(graph: nx.Graph, mode: str, seed: int | None) -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    if mode == "kamada":
        return nx.kamada_kawai_layout(graph)
    return nx.spring_layout(graph, seed=seed or 42)


def _export_graph(
    mapper: RegimeAwareMapper,
    cov_slice: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    layout_mode: str,
    dpi: int,
    seed: int | None,
) -> tuple[Path, Path]:
    json_path, png_path = mapper.export_graph(output_dir / prefix)

    # overwrite PNG with nicer layout if requested
    try:
        graph = mapper.graph_
        if graph is None:
            return json_path, png_path
        fig, ax = plt.subplots(figsize=(6, 5))
        layout = _layout(graph, layout_mode, seed)
        sizes = [max(80, graph.nodes[n].get("size", 1) * 80) for n in graph.nodes]
        weights = [
            cov_slice.loc[m, m] if isinstance(m, str) and m in cov_slice.index else 1.0
            for m in graph.nodes
        ]
        norm_weights = np.linspace(0.2, 0.8, len(weights)) if weights else []
        nx.draw_networkx(
            graph,
            pos=layout,
            ax=ax,
            with_labels=True,
            node_size=sizes,
            node_color=norm_weights,
            cmap="viridis",
            edge_color="#555555",
            linewidths=0.8,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to redraw graph layout: {exc}")
    return json_path, png_path


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    dates_cfg = cfg.get("dates", {})
    start = pd.Timestamp(args.start_date or dates_cfg.get("start"))
    end = pd.Timestamp(args.end_date or dates_cfg.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Configuration must define dates.start and dates.end")

    panel = get_panel(start, end)
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must have MultiIndex with date level")

    prices = panel["close"].unstack("asset").sort_index()
    returns = prices.pct_change().dropna(how="all")

    mapper_cfg = cfg.get("tda_mapper", {}) or {}
    mapper, mapper_params = _prepare_mapper(args, mapper_cfg)
    lookback = args.lookback or int(mapper_cfg.get("lookback", 252))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = _resolve_dates(prices, args.dates, args.frequency)
    saved = []

    for date in dates:
        if date not in returns.index:
            continue
        window_returns = returns.loc[:date].tail(lookback)
        if window_returns.empty or window_returns.shape[1] < 2:
            continue
        try:
            mapper.fit(
                window_returns,
                regime_value=None,
            )
        except Exception as exc:
            print(f"[WARN] Mapper fit failed on {date.date()}: {exc}")
            continue

        prefix = f"mapper_{date:%Y%m%d}_nc{mapper_params['n_cubes']}_ov{mapper_params['overlap']:.2f}"
        json_path, png_path = _export_graph(
            mapper,
            window_returns.cov(),
            output_dir,
            prefix,
            args.layout,
            args.dpi,
            mapper_params.get("random_state"),
        )

        node_metrics = mapper.metrics_.node_metrics.reset_index().to_dict(orient="records")
        payload = {
            "date": date.isoformat(),
            "lookback": lookback,
            "params": mapper_params,
            "summary": mapper.metrics_.summary,
            "node_metrics": node_metrics,
        }
        (output_dir / f"{prefix}.metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if args.verbose:
            print(f"[INFO] Exported Mapper for {date.date()} -> {png_path.name}")
        saved.append(
            {
                "date": date,
                "png": png_path,
                "json": json_path,
                "metrics": output_dir / f"{prefix}.metrics.json",
            }
        )

    if not saved:
        print("No Mapper exports were generated.")
    else:
        index_path = output_dir / "exports_index.csv"
        pd.DataFrame(
            [
                {
                    "date": entry["date"],
                    "png": entry["png"].name,
                    "json": entry["json"].name,
                    "metrics": entry["metrics"].name,
                }
                for entry in saved
            ]
        ).to_csv(index_path, index=False)
        print(f"Wrote {len(saved)} Mapper exports to {output_dir}")


if __name__ == "__main__":
    main()

