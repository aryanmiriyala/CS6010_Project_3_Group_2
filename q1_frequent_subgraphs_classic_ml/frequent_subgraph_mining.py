import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple
from gspan_mining import gSpan

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import load_mutag
from q1_frequent_subgraphs_classic_ml.graph_utils import extract_edges, extract_node_labels


@dataclass
class GraphRecord:
    """Lightweight graph container that mirrors the info gSpan expects."""

    gid: int
    label: int
    node_labels: List[int]
    edges: List[Tuple[int, int, int]]


class PatternRecorder(gSpan):
    """
    Custom gSpan subclass that captures the mined patterns without printing to stdout.
    """

    def __init__(self, database_file_name: str, min_support: int):
        super().__init__(
            database_file_name=database_file_name,
            min_support=min_support,
            verbose=False,
            visualize=False,
            where=True,
        )
        self.patterns: List[Dict] = []

    def _capture_graph_repr(self, graph_obj):
        vertices = []
        for vid in sorted(graph_obj.vertices.keys(), key=int):
            vlb = graph_obj.vertices[vid].vlb
            vertices.append({"id": int(vid), "label": int(vlb)})

        edges = set()
        for frm, vertex in graph_obj.vertices.items():
            for to, edge in vertex.edges.items():
                frm_int, to_int = int(frm), int(to)
                if self._is_undirected and frm_int > to_int:
                    continue
                edges.add(
                    (
                        frm_int,
                        to_int,
                        int(edge.elb),
                    )
                )

        return {"vertices": vertices, "edges": sorted(edges)}

    def _report_size1(self, g, support):
        if self._min_num_vertices > 1:
            return
        self.patterns.append(
            {
                "num_vertices": 1,
                "support": support,
                "graph": self._capture_graph_repr(g),
                "where": [],
            }
        )

    def _report(self, projected):
        self._frequent_subgraphs.append(copy.copy(self._DFScode))
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return

        g = self._DFScode.to_graph(
            gid=0,
            is_undirected=self._is_undirected,
        )

        self.patterns.append(
            {
                "num_vertices": self._DFScode.get_num_vertices(),
                "support": self._support,
                "graph": self._capture_graph_repr(g),
                "where": sorted({p.gid for p in projected}),
            }
        )


def build_graph_records(splits) -> Dict[int, List[GraphRecord]]:
    """
    Convert PyG Data objects into GraphRecord instances separated by class label.
    """

    graphs_by_label: Dict[int, List[GraphRecord]] = {}
    for gid, graph in enumerate(splits.train):
        label = int(graph.y.item())
        node_labels = extract_node_labels(graph)
        edges = extract_edges(graph)
        graphs_by_label.setdefault(label, []).append(
            GraphRecord(gid=gid, label=label, node_labels=node_labels, edges=edges)
        )
    return graphs_by_label


def write_gspan_file(graphs: Sequence[GraphRecord], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for graph in graphs:
            f.write(f"t # {graph.gid}\n")
            for vid, label in enumerate(graph.node_labels):
                f.write(f"v {vid} {label}\n")
            for frm, to, elb in graph.edges:
                f.write(f"e {frm} {to} {elb}\n")
        f.write("t # -1\n")


def mine_patterns(database_path: Path, min_support_count: int) -> Tuple[List[Dict], float]:
    miner = PatternRecorder(str(database_path), min_support=min_support_count)
    start = time.time()
    miner.run()
    runtime = time.time() - start
    return miner.patterns, runtime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frequent subgraph mining + classic ML pipeline for MUTAG (Q1)."
    )
    parser.add_argument(
        "--support-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
        help="Minimum support ratios (fraction of class-specific training graphs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory to store mined patterns and intermediate artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MUTAG dataset and preparing graph splits...")
    dataset, splits, _ = load_mutag(batch_size=1, shuffle=False)
    print(f"Loaded {len(splits.train)} training graphs across {dataset.num_classes} classes.")
    graphs_by_label = build_graph_records(splits)
    for label, graphs in graphs_by_label.items():
        print(f"  Class {label}: {len(graphs)} training graphs")

    summary = []
    for support_ratio in args.support_thresholds:
        print(f"\n=== Mining with support ratio {support_ratio:.2f} ===")
        threshold_entry = {"support_ratio": support_ratio, "classes": {}}
        for label, graphs in graphs_by_label.items():
            min_support = max(1, math.ceil(support_ratio * len(graphs)))
            class_dir = args.output_dir / f"class_{label}" / f"support_{support_ratio:.2f}"
            class_dir.mkdir(parents=True, exist_ok=True)

            db_path = class_dir / "graphs.gspan"
            print(
                f"Writing gSpan database for class {label} to "
                f"{db_path.relative_to(REPO_ROOT)} (min_support={min_support})"
            )
            write_gspan_file(graphs, db_path)

            print(f"Running gSpan for class {label} (support ratio {support_ratio:.2f})...")
            patterns, runtime = mine_patterns(db_path, min_support)
            output_json = class_dir / "patterns.json"
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "class_label": label,
                        "support_ratio": support_ratio,
                        "min_support_count": min_support,
                        "num_graphs": len(graphs),
                        "patterns": patterns,
                        "runtime_sec": runtime,
                    },
                    f,
                    indent=2,
                )

            threshold_entry["classes"][label] = {
                "num_graphs": len(graphs),
                "min_support_count": min_support,
                "num_patterns": len(patterns),
                "runtime_sec": runtime,
                "artifacts": str(class_dir.relative_to(args.output_dir)),
            }
            print(
                f"Completed mining for class {label} | "
                f"patterns: {len(patterns)} | runtime: {runtime:.2f}s | "
                f"saved to {output_json.relative_to(REPO_ROOT)}"
            )
        summary.append(threshold_entry)

    summary_path = args.output_dir / "mining_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved mining summary to {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
