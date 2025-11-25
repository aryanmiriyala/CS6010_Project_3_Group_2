import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import load_mutag
from q1_frequent_subgraphs_classic_ml.graph_utils import pyg_data_to_nx

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURES_DIR = Path(__file__).resolve().parent / "features"
TOP_K_PER_CLASS: int | None = 50  # set to None or 0 to keep all patterns


def load_patterns(artifacts_dir: Path, support_ratio: float, top_k: int | None) -> List[Dict]:
    selected = []
    ratio_str = f"{support_ratio:.2f}"
    for label in sorted(p.name for p in artifacts_dir.glob("class_*") if p.is_dir()):
        class_label = int(label.split("_")[1])
        patterns_path = artifacts_dir / label / f"support_{ratio_str}" / "patterns.json"
        if not patterns_path.exists():
            raise FileNotFoundError(f"Missing patterns file: {patterns_path}")
        payload = json.loads(patterns_path.read_text())
        patterns = sorted(payload["patterns"], key=lambda p: p["support"], reverse=True)
        trimmed = patterns if top_k in (None, 0) else patterns[:top_k]
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


def discover_support_ratios(artifacts_dir: Path) -> List[float]:
    ratios: Set[float] = set()
    for class_dir in artifacts_dir.glob("class_*"):
        for support_dir in class_dir.glob("support_*"):
            try:
                ratios.add(float(support_dir.name.split("_")[1]))
            except ValueError:
                continue
    return sorted(ratios)


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset, splits, _ = load_mutag(batch_size=1, shuffle=False)
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

    support_ratios = discover_support_ratios(ARTIFACTS_DIR)
    if not support_ratios:
        raise RuntimeError(f"No support directories found in {ARTIFACTS_DIR}")

    for support_ratio in support_ratios:
        print(f"\n=== Constructing features for support ratio {support_ratio:.2f} ===")
        try:
            patterns = load_patterns(ARTIFACTS_DIR, support_ratio, TOP_K_PER_CLASS)
        except FileNotFoundError as exc:
            print(f"Skipping support ratio {support_ratio:.2f}: {exc}")
            continue
        if not patterns:
            print(f"No patterns found for support ratio {support_ratio:.2f}, skipping.")
            continue

        pattern_graphs = [pattern_to_nx(p) for p in patterns]
        support_dir = FEATURES_DIR / f"support_{support_ratio:.2f}"
        support_dir.mkdir(parents=True, exist_ok=True)

        for split_name, graphs in datasets.items():
            feats = compute_features(graphs, pattern_graphs)
            np.savez(
                support_dir / f"{split_name}_features.npz",
                features=feats,
                labels=np.array(labels[split_name], dtype=np.int64),
            )

        metadata = {
            "support_ratio": support_ratio,
            "top_k_per_class": TOP_K_PER_CLASS,
            "num_patterns": len(patterns),
            "pattern_metadata": patterns,
        }
        with open(support_dir / "feature_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved features for support ratio {support_ratio:.2f} to {support_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
