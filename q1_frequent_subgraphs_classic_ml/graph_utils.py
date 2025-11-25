from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import torch


def extract_node_labels(graph) -> List[int]:
    """Return integer labels for each node in a PyG Data object."""
    if graph.x is None:
        return list(range(graph.num_nodes))

    if graph.x.dim() == 1:
        return graph.x.long().tolist()

    return torch.argmax(graph.x, dim=1).long().tolist()


def extract_edges(graph, undirected: bool = True) -> List[Tuple[int, int, int]]:
    """Return (src, dst, label) tuples for each edge, with optional undirected deduping."""
    edge_index = graph.edge_index.t().tolist()
    edge_attrs = graph.edge_attr if getattr(graph, "edge_attr", None) is not None else None
    edges = set()
    for idx, (src, dst) in enumerate(edge_index):
        if src == dst:
            continue
        frm, to = (src, dst) if not undirected or src <= dst else (dst, src)
        if edge_attrs is not None:
            if edge_attrs.dim() == 1:
                label = int(edge_attrs[idx].item())
            else:
                label = int(torch.argmax(edge_attrs[idx]).item())
        else:
            label = 0
        edges.add((frm, to, label))
    return sorted(edges)


def pyg_data_to_nx(graph) -> nx.Graph:
    """Convert a PyG Data object into a labeled NetworkX graph."""
    G = nx.Graph()
    node_labels = extract_node_labels(graph)
    for idx, label in enumerate(node_labels):
        G.add_node(idx, label=int(label))

    for frm, to, label in extract_edges(graph):
        G.add_edge(frm, to, label=int(label))

    return G
