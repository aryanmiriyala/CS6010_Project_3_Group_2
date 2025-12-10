#!/usr/bin/env python3
"""Extract classical motif importances from Q1 feature artifacts for Q4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC, SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
Q1_ROOT = REPO_ROOT / "q1_frequent_subgraphs_classic_ml"
FEATURES_ROOT = Q1_ROOT / "features"
RESULTS_ROOT = Q1_ROOT / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize classical motif importances using Q1 feature artifacts",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="Seed IDs to inspect (default: 0 1 2)",
    )
    parser.add_argument(
        "--supports",
        type=float,
        nargs="*",
        default=[0.10, 0.20, 0.30, 0.40],
        help="Support ratios to inspect (default: 0.10 0.20 0.30 0.40)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many motifs per model/seed/support to record",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "classic_motif_importances.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["features"], data["labels"]


def load_feature_bundle(support_dir: Path) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    bundle: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        split_path = support_dir / f"{split}_features.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        bundle[split] = load_split(split_path)
    return bundle


def read_best_params(results_csv: Path, model_name: str) -> Dict | None:
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    subset = df[df["model"] == model_name]
    if subset.empty:
        return None
    best = subset.sort_values("val_accuracy", ascending=False).iloc[0]
    return json.loads(best["params"])


def train_random_forest(X: np.ndarray, y: np.ndarray, params: Dict) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth"),
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_linear_svm(X: np.ndarray, y: np.ndarray, params: Dict) -> LinearSVC:
    dual_flag = X.shape[0] < X.shape[1]
    model = LinearSVC(
        C=float(params.get("C", 1.0)),
        max_iter=600000,
        dual=dual_flag,
        tol=5e-3,
    )
    model.fit(X, y)
    return model


def train_rbf_svm(X: np.ndarray, y: np.ndarray, params: Dict) -> SVC:
    model = SVC(
        C=float(params.get("C", 1.0)),
        gamma=params.get("gamma", "scale"),
        kernel="rbf",
        probability=True,
    )
    model.fit(X, y)
    return model


def build_feature_names(pattern_metadata: List[Dict]) -> List[str]:
    names = []
    for idx, pattern in enumerate(pattern_metadata):
        names.append(
            f"pattern_{idx}_class{pattern['class_label']}_supp{pattern['support']}"
        )
    return names


def main() -> None:
    args = parse_args()
    rows: List[Dict] = []

    for seed in args.seeds:
        for support in args.supports:
            support_dir = FEATURES_ROOT / f"seed_{seed}" / f"support_{support:.2f}"
            if not support_dir.exists():
                print(f"[warn] Missing features for seed={seed} support={support:.2f}; skipping")
                continue

            config_path = support_dir / "feature_config.json"
            metadata = json.loads(config_path.read_text())
            feature_names = build_feature_names(metadata["pattern_metadata"])

            bundle = load_feature_bundle(support_dir)
            X_train = np.concatenate([bundle["train"][0], bundle["val"][0]], axis=0)
            y_train = np.concatenate([bundle["train"][1], bundle["val"][1]], axis=0)

            results_csv = RESULTS_ROOT / f"seed_{seed}" / f"classic_ml_support_{support:.2f}.csv"

            # RandomForest importances
            rf_params = read_best_params(results_csv, "RandomForest")
            if rf_params:
                rf_model = train_random_forest(X_train, y_train, rf_params)
                rf_importances = rf_model.feature_importances_
                top_idx = np.argsort(rf_importances)[::-1][: args.top_k]
                for rank, idx in enumerate(top_idx, start=1):
                    pattern = metadata["pattern_metadata"][idx]
                    rows.append(
                        {
                            "seed": seed,
                            "support_ratio": round(support, 2),
                            "model": "RandomForest",
                            "feature_rank": rank,
                            "feature_index": idx,
                            "feature_name": feature_names[idx],
                            "class_label": pattern["class_label"],
                            "pattern_support_count": pattern["support"],
                            "importance": float(rf_importances[idx]),
                            "coefficient": np.nan,
                            "graph": json.dumps(pattern["graph"]),
                        }
                    )
            else:
                print(
                    f"[warn] No RandomForest results for seed={seed} support={support:.2f}; skipping RF",
                )

            # Linear SVM coefficients (if available)
            svm_params = read_best_params(results_csv, "LinearSVM")
            if svm_params:
                try:
                    svm_model = train_linear_svm(X_train, y_train, svm_params)
                    coefs = svm_model.coef_.ravel()
                    order = np.argsort(np.abs(coefs))[::-1][: args.top_k]
                    for rank, idx in enumerate(order, start=1):
                        pattern = metadata["pattern_metadata"][idx]
                        rows.append(
                            {
                                "seed": seed,
                                "support_ratio": round(support, 2),
                                "model": "LinearSVM",
                                "feature_rank": rank,
                                "feature_index": idx,
                                "feature_name": feature_names[idx],
                                "class_label": pattern["class_label"],
                                "pattern_support_count": pattern["support"],
                                "importance": float(abs(coefs[idx])),
                                "coefficient": float(coefs[idx]),
                                "graph": json.dumps(pattern["graph"]),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] Failed to train/extract LinearSVM for seed={seed} support={support:.2f}: {exc}",
                    )
            else:
                print(
                    f"[warn] No LinearSVM results for seed={seed} support={support:.2f}; skipping SVM",
                )

            # RBFSVM permutation importance
            rbf_params = read_best_params(results_csv, "RBFSVM")
            if rbf_params:
                try:
                    rbf_model = train_rbf_svm(X_train, y_train, rbf_params)
                    perm = permutation_importance(
                        rbf_model,
                        bundle["val"][0],
                        bundle["val"][1],
                        n_repeats=5,
                        random_state=42,
                        n_jobs=1,
                    )
                    importance_scores = perm.importances_mean
                    top_idx = np.argsort(np.abs(importance_scores))[::-1][: args.top_k]
                    for rank, idx in enumerate(top_idx, start=1):
                        pattern = metadata["pattern_metadata"][idx]
                        rows.append(
                            {
                                "seed": seed,
                                "support_ratio": round(support, 2),
                                "model": "RBFSVM",
                                "feature_rank": rank,
                                "feature_index": idx,
                                "feature_name": feature_names[idx],
                                "class_label": pattern["class_label"],
                                "pattern_support_count": pattern["support"],
                                "importance": float(importance_scores[idx]),
                                "coefficient": np.nan,
                                "graph": json.dumps(pattern["graph"]),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] Failed RBFSVM permutation importance seed={seed} support={support:.2f}: {exc}",
                    )

            else:
                print(
                    f"[warn] No RBFSVM results for seed={seed} support={support:.2f}; skipping RBFSVM",
                )

    if not rows:
        raise SystemExit("No importances were extracted. Check inputs.")

    df = pd.DataFrame(rows)
    df.sort_values(["seed", "support_ratio", "model", "feature_rank"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved classical motif importances to {args.output}")

    # Write per-seed files mirroring the GNN structure
    for (seed, model_name), seed_df in df.groupby(["seed", "model"]):
        seed_dir = OUTPUT_DIR / model_name / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_path = seed_dir / args.output.name
        seed_df.to_csv(seed_path, index=False)
        print(f"{model_name} seed {seed} motif importances -> {seed_path}")


if __name__ == "__main__":
    main()
