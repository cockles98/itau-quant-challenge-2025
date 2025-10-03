#!/usr/bin/env python
"""Export Mapper graphs and embeddings for a specific asset/date."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from dataio.config import load_config
from dataio.loaders import get_panel
from features import TFIParams, mapper_for_asset


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Type {obj.__class__.__name__} is not JSON serializable")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and persist TDA Mapper graphs for a given asset/date."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to configuration YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--asset",
        default=None,
        help="Ticker/asset column to inspect. Default: first column found.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Final date (YYYY-MM-DD) to anchor the observation window.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tda_maps",
        help="Directory to save artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix for output files.",
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
        help="Figure resolution in DPI (default: %(default)s)",
    )
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-cubes", type=int, default=None, dest="n_cubes")
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=None, dest="min_samples")
    parser.add_argument("--window", type=int, default=None)
    return parser.parse_args()


def _apply_overrides(params: TFIParams, args: argparse.Namespace) -> TFIParams:
    data = {
        "delay": args.delay,
        "dim": args.dim,
        "n_cubes": args.n_cubes,
        "overlap": args.overlap,
        "epsilon": None if args.epsilon is None else args.epsilon,
        "min_samples": args.min_samples,
        "window": args.window,
    }
    clean = {k: v for k, v in data.items() if v is not None}
    if not clean:
        return params
    return replace(params, **clean)


def _choose_layout(graph: nx.Graph, mode: str) -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    if mode == "kamada":
        return nx.kamada_kawai_layout(graph)
    return nx.spring_layout(graph, seed=42)


def _ensure_asset(prices: pd.DataFrame, asset: str | None) -> str:
    columns = list(prices.columns)
    if not columns:
        raise ValueError("Price panel is empty")
    if asset is None:
        return columns[0]
    if asset not in columns:
        raise KeyError(f"Asset '{asset}' not found. Available: {columns[:5]} ...")
    return asset


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    dates_cfg = cfg.get("dates", {})
    start = pd.Timestamp(dates_cfg.get("start"))
    end = pd.Timestamp(dates_cfg.get("end"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Config must include dates.start and dates.end")

    panel = get_panel(start, end)
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must have a MultiIndex with a date level")

    prices = panel["close"].unstack("asset").sort_index()
    asset = _ensure_asset(prices, args.asset)

    end_date = pd.Timestamp(args.date) if args.date else None

    windows_cfg = cfg.get("windows", {}) or {}
    vol_window = int(windows_cfg.get("vol_window", 126))
    params = TFIParams.from_config(cfg, vol_window_fallback=vol_window)
    params = _apply_overrides(params, args)

    graph, embedding, metadata = mapper_for_asset(
        prices,
        params,
        asset,
        end_date=end_date,
        return_embedding=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_end = metadata["window"]["end"].split("T")[0]
    prefix = args.prefix.strip() + "_" if args.prefix else ""
    stem = f"{prefix}{asset}_{window_end}_nc{params.n_cubes}_ov{params.overlap:.2f}"
    figure_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_graph, ax_embed = axes

    layout = _choose_layout(graph, args.layout)
    node_sizes = [max(40.0, graph.nodes[n].get("size", 1) * 20.0) for n in graph.nodes]
    node_colors = np.linspace(0.2, 0.8, len(node_sizes)) if node_sizes else []

    ax_graph.set_title(f"Mapper graph | {asset}")
    if graph.number_of_nodes():
        nx.draw_networkx(
            graph,
            pos=layout,
            ax=ax_graph,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            cmap="viridis",
            edge_color="#555555",
        )
    else:
        ax_graph.text(0.5, 0.5, "Graph vazio", ha="center", va="center")
    ax_graph.axis("off")

    ax_embed.set_title("Takens embedding (primeiras 2 dims)")
    if embedding.size:
        x = embedding[:, 0]
        y = embedding[:, 1] if embedding.shape[1] > 1 else np.zeros_like(x)
        ax_embed.scatter(x, y, s=15, alpha=0.7, c=np.linspace(0, 1, len(x)), cmap="viridis")
    ax_embed.set_xlabel("dim 1")
    ax_embed.set_ylabel("dim 2")
    ax_embed.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=args.dpi)
    plt.close(fig)

    graph_payload = nx.readwrite.json_graph.node_link_data(graph, edges="links")
    payload = {
        "metadata": metadata,
        "params": {
            "delay": params.delay,
            "dim": params.dim,
            "n_cubes": params.n_cubes,
            "overlap": params.overlap,
            "epsilon": params.epsilon,
            "min_samples": params.min_samples,
            "window": params.window,
        },
        "graph": graph_payload,
        "embedding_shape": embedding.shape,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    print(f"Saved figure to {figure_path}")
    print(f"Saved graph payload to {json_path}")


if __name__ == "__main__":
    main()



