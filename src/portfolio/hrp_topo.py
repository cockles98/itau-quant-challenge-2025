from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd

__all__ = ["topo_seriation_from_graph"]


def topo_seriation_from_graph(
    cov: pd.DataFrame,
    graph: nx.Graph | None,
    *,
    mapper_graph: nx.Graph | Dict | None = None,
    mapper_centrality: Dict[str, float] | None = None,
) -> List[str]:
    """Derive an asset ordering for HRP given covariance and optional Mapper graph.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix indexed by asset identifiers.
    graph : nx.Graph | None
        Baseline asset graph (e.g., minimum spanning tree) used for fallback seriation.
    mapper_graph : nx.Graph | dict | None, optional
        Mapper graph produced by the topological pipeline. When provided, a DFS per
        connected component is executed prioritising nodes with higher centrality.
    mapper_centrality : dict, optional
        Pre-computed centrality scores per Mapper node. When omitted, the scores are
        inferred from node attributes or standard centrality measures.
    """

    if cov.empty:
        return []

    assets = list(cov.columns)

    mapper_order: List[str] | None = None
    mapper_obj = _coerce_mapper_graph(mapper_graph)
    if mapper_obj is not None and mapper_obj.number_of_nodes() > 0:
        mapper_order = _order_from_mapper_graph(
            cov,
            mapper_obj,
            assets,
            mapper_centrality=mapper_centrality,
        )
    if mapper_order:
        return mapper_order

    return _fallback_seriation(cov, graph, assets)


# --------------------------------------------------------------------------- #
# Mapper-aware seriation

def _order_from_mapper_graph(
    cov: pd.DataFrame,
    mapper_graph: nx.Graph,
    assets: Sequence[str],
    *,
    mapper_centrality: Dict[str, float] | None = None,
) -> List[str]:
    nodes_with_assets = _collect_nodes_with_assets(mapper_graph, assets)
    if not nodes_with_assets:
        return []

    subgraph = mapper_graph.subgraph(nodes_with_assets).copy()
    centrality = _extract_centrality(subgraph, mapper_centrality)
    if not centrality:
        centrality = _compute_default_centrality(subgraph)

    components = list(nx.connected_components(subgraph))

    def _component_priority(component: Iterable[str]) -> float:
        return max(centrality.get(node, float("-inf")) for node in component)

    components.sort(key=_component_priority, reverse=True)

    visited_nodes: List[str] = []
    for component in components:
        component_nodes = list(component)
        start_node = max(component_nodes, key=lambda node: centrality.get(node, float("-inf")))
        component_subgraph = subgraph.subgraph(component_nodes)
        visited_nodes.extend(
            _dfs_with_priority(component_subgraph, start_node, centrality)
        )

    ordered_assets: List[str] = []
    seen_assets: set[str] = set()
    len_assets = len(assets)

    for node in visited_nodes:
        members = _resolve_members(mapper_graph.nodes[node], assets, node, len_assets)
        if not members and node in assets:
            members = [node]
        for asset in members:
            if asset not in seen_assets:
                ordered_assets.append(asset)
                seen_assets.add(asset)

    if not ordered_assets:
        return []

    for asset in assets:
        if asset not in seen_assets:
            ordered_assets.append(asset)

    return ordered_assets


def _coerce_mapper_graph(mapper_graph: nx.Graph | Dict | None) -> nx.Graph | None:
    if isinstance(mapper_graph, nx.Graph):
        return mapper_graph
    if isinstance(mapper_graph, dict):
        nodes = mapper_graph.get("nodes")
        links = mapper_graph.get("links")
        if not isinstance(nodes, dict):
            return None
        graph = nx.Graph()
        for node_id, attrs in nodes.items():
            if isinstance(attrs, dict):
                graph.add_node(node_id, **attrs)
            else:
                graph.add_node(node_id, members=attrs)
        if isinstance(links, dict):
            for node_id, neighbours in links.items():
                for neighbour in neighbours:
                    graph.add_edge(node_id, neighbour)
        return graph
    return None


def _collect_nodes_with_assets(
    mapper_graph: nx.Graph,
    assets: Sequence[str],
) -> List[str]:
    len_assets = len(assets)
    valid_nodes: List[str] = []
    for node, attrs in mapper_graph.nodes(data=True):
        members = _resolve_members(attrs, assets, node, len_assets)
        if members or node in assets:
            valid_nodes.append(node)
    return valid_nodes


def _resolve_members(
    attrs: Dict[str, object],
    assets: Sequence[str],
    fallback: str,
    len_assets: int,
) -> List[str]:
    raw_members = attrs.get("members") if isinstance(attrs, dict) else None
    if raw_members is None:
        return []
    members: List[str] = []
    for value in raw_members:
        if isinstance(value, str) and value in assets:
            members.append(value)
        else:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len_assets:
                members.append(assets[idx])
    if not members and fallback in assets:
        return [fallback]
    return members


def _extract_centrality(
    mapper_graph: nx.Graph,
    mapper_centrality: Dict[str, float] | None,
) -> Dict[str, float]:
    centrality: Dict[str, float] = {}
    if mapper_centrality:
        for node, value in mapper_centrality.items():
            try:
                centrality[node] = float(value)
            except (TypeError, ValueError):
                continue
        if centrality:
            return centrality

    for node, attrs in mapper_graph.nodes(data=True):
        if not isinstance(attrs, dict):
            continue
        raw = attrs.get("centrality")
        if isinstance(raw, dict):
            values = [
                float(raw[key])
                for key in ("degree", "eigen", "betweenness", "score")
                if isinstance(raw.get(key), (int, float))
            ]
            if values:
                centrality[node] = float(np.mean(values))
        elif isinstance(raw, (int, float)):
            centrality[node] = float(raw)
    if centrality:
        return centrality
    return {}


def _compute_default_centrality(mapper_graph: nx.Graph) -> Dict[str, float]:
    deg = nx.degree_centrality(mapper_graph)
    try:
        eigen = nx.eigenvector_centrality(mapper_graph, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eigen = {node: 0.0 for node in mapper_graph.nodes}
    bet = nx.betweenness_centrality(mapper_graph, normalized=True)
    combined: Dict[str, float] = {}
    for node in mapper_graph.nodes:
        values = [
            deg.get(node, 0.0),
            eigen.get(node, 0.0),
            bet.get(node, 0.0),
        ]
        combined[node] = float(np.mean(values))
    return combined


def _dfs_with_priority(
    graph: nx.Graph,
    start_node: str,
    centrality: Dict[str, float],
) -> List[str]:
    visited: List[str] = []
    seen: set[str] = set()
    stack: List[str] = [start_node]

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        neighbours = [nbr for nbr in graph.neighbors(node) if nbr not in seen]
        neighbours.sort(key=lambda nbr: centrality.get(nbr, float("-inf")))
        for neighbour in neighbours:
            stack.append(neighbour)
    return visited


# --------------------------------------------------------------------------- #
# Legacy fallback seriation

def _fallback_seriation(
    cov: pd.DataFrame,
    graph: nx.Graph | None,
    assets: Sequence[str],
) -> List[str]:
    if graph is None or graph.number_of_nodes() == 0:
        return list(assets)

    nodes = [node for node in graph.nodes if node in assets]
    if not nodes:
        return list(assets)

    subgraph = graph.subgraph(nodes).copy()
    order: List[str] = []

    def component_score(component_nodes: Sequence[str]) -> float:
        subcov = cov.loc[component_nodes, component_nodes]
        return float(subcov.values.mean())

    components = sorted(
        nx.connected_components(subgraph),
        key=lambda comp: component_score(list(comp)),
    )

    for component in components:
        comp_nodes = list(component)
        comp_subgraph = subgraph.subgraph(comp_nodes)
        start_node = max(
            comp_nodes,
            key=lambda node: float(cov.loc[node, node]) if node in cov.index else float("-inf"),
        )
        order.extend(list(nx.dfs_preorder_nodes(comp_subgraph, source=start_node)))

    serialised: List[str] = []
    seen: set[str] = set()
    for node in order:
        if node not in seen:
            serialised.append(node)
            seen.add(node)

    for asset in assets:
        if asset not in seen:
            serialised.append(asset)

    return serialised

