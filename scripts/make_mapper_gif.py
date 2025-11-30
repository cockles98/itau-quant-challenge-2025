from __future__ import annotations

"""Build a Mapper animation GIF from saved graph PNGs or graph JSONs."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import networkx as nx

plt.switch_backend("Agg")


def _collect_png_frames(root: Path, stride: int = 1, limit: int | None = None) -> List[Path]:
    """Return ordered PNG paths under artifacts/tda/mapper_* directories."""
    mapper_dirs = sorted(root.glob("mapper_*"))
    frames: List[Path] = []
    for folder in mapper_dirs:
        png_path = folder / "mapper_graph.png"
        json_path = folder / "mapper_graph.json"
        if png_path.exists() and json_path.exists():
            try:
                import json

                data = json.loads(json_path.read_text())
                nodes = data.get("graph", {}).get("nodes", {}) or data.get("nodes", {})
                if len(nodes) == 0:
                    continue
            except Exception:
                pass
            frames.append(png_path)
    if stride > 1:
        frames = frames[::stride]
    if limit is not None:
        frames = frames[:limit]
    return frames


def _collect_graph_jsons(root: Path) -> List[Path]:
    """Return Mapper graph JSONs (non-empty) from artifacts/mapper."""
    jsons: List[Path] = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("graph_"):
            jsons.append(path)
    return jsons


def _draw_graph(json_path: Path, target_png: Path) -> None:
    """Render a Mapper graph JSON to PNG."""
    import json
    import re

    data = json.loads(json_path.read_text())
    nodes = data.get("graph", {}).get("nodes") or data.get("nodes") or {}
    links = data.get("graph", {}).get("links") or data.get("links") or {}
    if not nodes:
        return

    g = nx.Graph()
    for node_id, members in nodes.items():
        size = len(members) if isinstance(members, list) else 1
        g.add_node(node_id, size=size)
    for left, neighbours in links.items():
        for right in neighbours:
            g.add_edge(left, right)

    if g.number_of_nodes() == 0:
        return

    # Visual styling: size by cluster size, color by degree centrality.
    deg_cent = nx.degree_centrality(g)
    sizes = [max(120.0, 40.0 * g.nodes[n].get("size", 1)) for n in g.nodes]
    colors = [deg_cent.get(n, 0.0) for n in g.nodes]
    pos = nx.spring_layout(g, seed=42)

    plt.figure(figsize=(8, 6), facecolor="#0b1021")
    ax = plt.gca()
    ax.set_facecolor("#0b1021")
    nodes_plot = nx.draw_networkx_nodes(
        g,
        pos,
        node_size=sizes,
        node_color=colors,
        cmap="plasma",
        alpha=0.9,
    )
    nx.draw_networkx_edges(g, pos, edge_color="#cfd8e3", width=1.0, alpha=0.6)
    nx.draw_networkx_labels(g, pos, font_size=8, font_color="#f7f9fc")
    cbar = plt.colorbar(nodes_plot, shrink=0.8, pad=0.02)
    cbar.set_label("Centralidade (grau)", color="#f7f9fc")
    cbar.ax.yaxis.set_tick_params(color="#f7f9fc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#f7f9fc")

    # Try to derive a readable date from folder name (mapper_YYYYMMDD)
    folder = json_path.parent.name
    match = re.search(r"(\d{8})", folder)
    if match:
        raw = match.group(1)
        date_label = f"{raw[6:8]}/{raw[4:6]}/{raw[0:4]}"
    else:
        # fallback: keep folder name
        date_label = folder
    plt.title(f"Topologia de mercado (Mapper) - {date_label}", color="#f7f9fc")
    plt.axis("off")
    target_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_png, dpi=160, facecolor="#0b1021")
    plt.close()


def _frames_from_graph_jsons(graph_root: Path) -> List[Path]:
    """Render graph JSONs into persistent PNG frames."""
    jsons = _collect_graph_jsons(graph_root)
    if not jsons:
        return []
    out_dir = Path("artifacts/cache/mapper_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[Path] = []
    for idx, json_path in enumerate(jsons):
        png_path = out_dir / f"graph_frame_{idx:03d}.png"
        _draw_graph(json_path, png_path)
        if png_path.exists():
            frames.append(png_path)
    return frames


def build_gif(pngs: Iterable[Path], output: Path, duration: float = 0.8) -> None:
    """Assemble PNG frames into a GIF.

    The CLI accepts *duration* in seconds, but imageio expects GIF durations in
    milliseconds; we convert internally.
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - convenience wrapper
        raise SystemExit("Install imageio to build the Mapper GIF: pip install imageio") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    frames = [iio.imread(path) for path in pngs]
    if not frames:
        raise SystemExit("No mapper_graph.png frames found under artifacts/tda")
    duration_ms = max(int(round(duration * 1000)), 20)  # GIF uses centiseconds; clamp minimum
    iio.imwrite(output, frames, duration=duration_ms, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GIF from Mapper graph PNG frames.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/tda"),
        help="Directory containing mapper_* subfolders with mapper_graph.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/readme_assets/mapper_tda.gif"),
        help="Output GIF path.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame to reduce size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of frames.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.8,
        help="Frame duration (seconds) in the GIF.",
    )
    parser.add_argument(
        "--render-json",
        action="store_true",
        help="Ignore existing PNGs and render frames from mapper_graph.json instead (better styling/labels).",
    )
    args = parser.parse_args()

    frames: List[Path] = []
    if args.render_json:
        # Force rendering from JSON in artifacts/tda
        json_paths = sorted(Path(args.root).glob("mapper_*/mapper_graph.json"))
        for idx, json_path in enumerate(json_paths):
            png_path = Path("artifacts/cache/mapper_frames") / f"mapper_frame_{idx:03d}.png"
            _draw_graph(json_path, png_path)
            if png_path.exists():
                frames.append(png_path)
    else:
        frames = _collect_png_frames(args.root, stride=max(1, args.stride), limit=args.limit)
        if len(frames) < 2:
            # fallback to graph JSONs under artifacts/mapper
            graph_frames = _frames_from_graph_jsons(Path("artifacts/mapper"))
            if graph_frames:
                frames = graph_frames
    if not frames:
        raise SystemExit(
            f"No mapper frames found. Ensure mapper_graph.png exists under {args.root} or graph_*.json under artifacts/mapper."
        )
    build_gif(frames, args.output, duration=args.duration)
    print(f"GIF generated: {args.output} ({len(frames)} frames)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
