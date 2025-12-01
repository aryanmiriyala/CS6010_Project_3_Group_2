import argparse
import json
import time
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
FEATURES_SEEDS = [0, 1, 2]
DEFAULT_BASE_TOP_K_PER_CLASS: int | None = 50
SUPPORT_RATIO_TOP_K_MULTIPLIERS: Dict[str, float] = {
    "0.10": 8.0,
    "0.20": 4.0,
    "0.30": 2.0,
    "0.40": 1.0,
}


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


def compute_features(
    graphs: List[nx.Graph],
    pattern_graphs: List[nx.Graph],
    binary_features: bool,
) -> np.ndarray:
    node_match = isomorphism.categorical_node_match("label", None)
    edge_match = isomorphism.categorical_edge_match("label", None)
    features = np.zeros((len(graphs), len(pattern_graphs)), dtype=np.float32)
    for idx, pattern in enumerate(pattern_graphs):
        for g_idx, graph in enumerate(graphs):
            matcher = isomorphism.GraphMatcher(
                graph, pattern, node_match=node_match, edge_match=edge_match
            )
            if binary_features:
                if matcher.subgraph_is_isomorphic():
                    features[g_idx, idx] = 1.0
            else:
                count = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
                if count:
                    features[g_idx, idx] = float(count)
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


def format_ratio_key(support_ratio: float) -> str:
    return f"{support_ratio:.2f}"


def resolve_top_k(support_ratio: float, base_top_k: int | None, enable_scaling: bool) -> int | None:
    if base_top_k is None or base_top_k <= 0:
        return None
    multiplier = 1.0
    if enable_scaling:
        multiplier = SUPPORT_RATIO_TOP_K_MULTIPLIERS.get(format_ratio_key(support_ratio), 1.0)
    resolved = max(1, int(round(base_top_k * multiplier)))
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Construct classic ML feature matrices from mined motifs.")
    parser.add_argument(
        "--base-top-k-per-class",
        type=int,
        default=DEFAULT_BASE_TOP_K_PER_CLASS,
        help=(
            "How many motifs per class to keep before computing features. "
            "Use 0 to keep all motifs for a support ratio."
        ),
    )
    parser.add_argument(
        "--disable-ratio-scaling",
        action="store_true",
        help=(
            "Use the same --base-top-k-per-class for every support ratio instead of allocating "
            "more motifs to the lower thresholds."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Location of the mined patterns produced by frequent_subgraph_mining.py.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=FEATURES_DIR,
        help="Directory where the computed feature matrices will be written.",
    )
    parser.add_argument(
        "--binary-features",
        action="store_true",
        help="Store binary indicators instead of motif counts (default is counts).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    features_dir = args.features_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    support_seed_dirs = [
        p for p in artifacts_dir.glob("seed_*") if p.is_dir()
    ]
    if not support_seed_dirs:
        raise RuntimeError(f"No seed-specific artifact directories found in {artifacts_dir}")

    for seed_dir in sorted(support_seed_dirs):
        try:
            seed = int(seed_dir.name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping malformed seed directory: {seed_dir}")
            continue
        if seed not in FEATURES_SEEDS:
            continue

        print(f"\n=== Constructing features for seed {seed} ===")
        dataset, splits, _ = load_mutag(batch_size=1, shuffle=True, seed=seed)
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

        support_ratios = discover_support_ratios(seed_dir)
        if not support_ratios:
            print(f"No support directories found for seed {seed}, skipping.")
            continue

        for support_ratio in support_ratios:
            print(
                f"--- Constructing features for support ratio {support_ratio:.2f} (seed={seed}) ---"
            )
            top_k = resolve_top_k(
                support_ratio,
                args.base_top_k_per_class,
                enable_scaling=not args.disable_ratio_scaling,
            )
            try:
                patterns = load_patterns(seed_dir, support_ratio, top_k)
            except FileNotFoundError as exc:
                print(f"Skipping support ratio {support_ratio:.2f}: {exc}")
                continue
            if not patterns:
                print(f"No patterns found for support ratio {support_ratio:.2f}, skipping.")
                continue

            pattern_graphs = [pattern_to_nx(p) for p in patterns]
            support_dir = features_dir / f"seed_{seed}" / f"support_{support_ratio:.2f}"
            support_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.time()
            for split_name, graphs in datasets.items():
                feats = compute_features(
                    graphs, pattern_graphs, binary_features=args.binary_features
                )
                np.savez(
                    support_dir / f"{split_name}_features.npz",
                    features=feats,
                    labels=np.array(labels[split_name], dtype=np.int64),
                )
            runtime_sec = time.time() - start_time

            metadata = {
                "seed": seed,
                "support_ratio": support_ratio,
                "base_top_k_per_class": args.base_top_k_per_class,
                "scaled_top_k_per_class": top_k,
                "num_patterns": len(patterns),
                "feature_mode": "binary" if args.binary_features else "counts",
                "feature_construction_runtime_sec": runtime_sec,
                "pattern_metadata": patterns,
            }
            with open(support_dir / "feature_config.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(
                f"Saved features for seed {seed} support {support_ratio:.2f} to "
                f"{support_dir.relative_to(REPO_ROOT)}"
            )


if __name__ == "__main__":
    main()
