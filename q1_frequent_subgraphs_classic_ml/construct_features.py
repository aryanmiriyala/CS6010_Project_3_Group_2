import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import load_mutag
from q1_frequent_subgraphs_classic_ml.graph_utils import pyg_data_to_nx


def load_patterns(artifacts_dir: Path, support_ratio: float, top_k: int) -> List[Dict]:
    selected = []
    ratio_str = f"{support_ratio:.2f}"
    for label in sorted(p.name for p in artifacts_dir.glob("class_*") if p.is_dir()):
        class_label = int(label.split("_")[1])
        patterns_path = artifacts_dir / label / f"support_{ratio_str}" / "patterns.json"
        if not patterns_path.exists():
            raise FileNotFoundError(f"Missing patterns file: {patterns_path}")
        payload = json.loads(patterns_path.read_text())
        patterns = sorted(payload["patterns"], key=lambda p: p["support"], reverse=True)
        trimmed = patterns[:top_k] if top_k > 0 else patterns
        for patt in trimmed:
            selected.append(
                {
                    "class_label": class_label,
                    "support": patt["support"],
                    "graph": patt["graph"],
                }
            )
    return selected


def pattern_to_nx(pattern: Dict) -> nx.Graph:
    G = nx.Graph()
    for node in pattern["graph"]["vertices"]:
        G.add_node(int(node["id"]), label=int(node["label"]))
    for frm, to, elb in pattern["graph"]["edges"]:
        G.add_edge(int(frm), int(to), label=int(elb))
    return G


def compute_features(graphs: List[nx.Graph], pattern_graphs: List[nx.Graph]) -> np.ndarray:
    node_match = isomorphism.categorical_node_match("label", None)
    edge_match = isomorphism.categorical_edge_match("label", None)
    features = np.zeros((len(graphs), len(pattern_graphs)), dtype=np.float32)
    for idx, pattern in enumerate(pattern_graphs):
        for g_idx, graph in enumerate(graphs):
            matcher = isomorphism.GraphMatcher(graph, pattern, node_match=node_match, edge_match=edge_match)
            if matcher.subgraph_is_isomorphic():
                features[g_idx, idx] = 1.0
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Construct feature matrices from mined patterns.")
    parser.add_argument(
        "--support-ratio",
        type=float,
        default=0.3,
        help="Support ratio used when mining patterns (must match available artifacts).",
    )
    parser.add_argument(
        "--top-k-per-class",
        type=int,
        default=50,
        help="Number of top patterns (by support) to keep per class.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory containing mined pattern artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "features",
        help="Directory to save feature matrices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, splits, _ = load_mutag(batch_size=1, shuffle=False)
    patterns = load_patterns(args.artifacts_dir, args.support_ratio, args.top_k_per_class)
    if not patterns:
        raise RuntimeError("No patterns loaded. Ensure artifacts exist for the given support ratio.")

    pattern_graphs = [pattern_to_nx(p) for p in patterns]
    datasets = {
        "train": [pyg_data_to_nx(data) for data in splits.train],
        "val": [pyg_data_to_nx(data) for data in splits.val],
        "test": [pyg_data_to_nx(data) for data in splits.test],
    }
    labels = {
        "train": [int(data.y.item()) for data in splits.train],
        "val": [int(data.y.item()) for data in splits.val],
        "test": [int(data.y.item()) for data in splits.test],
    }

    for split_name, graphs in datasets.items():
        feats = compute_features(graphs, pattern_graphs)
        np.savez(
            args.output_dir / f"{split_name}_features.npz",
            features=feats,
            labels=np.array(labels[split_name], dtype=np.int64),
        )

    metadata = {
        "support_ratio": args.support_ratio,
        "top_k_per_class": args.top_k_per_class,
        "num_patterns": len(patterns),
        "pattern_metadata": patterns,
    }
    with open(args.output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved binary presence features to {args.output_dir}")


if __name__ == "__main__":
    main()
