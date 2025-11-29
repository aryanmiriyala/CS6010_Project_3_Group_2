import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC, SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = Path(__file__).resolve().parent / "features"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class FeatureSet:
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    metadata: Dict


def load_feature_set(support_ratio: float) -> FeatureSet:
    support_dir = FEATURES_DIR / f"support_{support_ratio:.2f}"
    if not support_dir.exists():
        raise FileNotFoundError(f"Features for support {support_ratio:.2f} not found at {support_dir}")

    def load_split(name: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(support_dir / f"{name}_features.npz")
        return data["features"], data["labels"]

    with open(support_dir / "feature_config.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_X, train_y = load_split("train")
    val_X, val_y = load_split("val")
    test_X, test_y = load_split("test")

    return FeatureSet(train_X, train_y, val_X, val_y, test_X, test_y, metadata)


def evaluate_model(model, X_val, y_val, X_test, y_test) -> Dict[str, float]:
    start = time.time()
    val_pred = model.predict(X_val)
    val_inference = time.time() - start

    start = time.time()
    test_pred = model.predict(X_test)
    test_inference = time.time() - start

    metrics = {
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_precision": precision_score(y_val, val_pred, average="macro", zero_division=0),
        "val_recall": recall_score(y_val, val_pred, average="macro", zero_division=0),
        "val_f1": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "val_inference_time_sec": val_inference,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, average="macro", zero_division=0),
        "test_recall": recall_score(y_test, test_pred, average="macro", zero_division=0),
        "test_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "test_inference_time_sec": test_inference,
    }
    return metrics


DEFAULT_RF_CONFIGS: Sequence[Dict] = (
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 20},
)
DEFAULT_SVM_CONFIGS: Sequence[Dict] = (
    {"type": "linear", "C": 0.1},
    {"type": "linear", "C": 1.0},
    {"type": "rbf", "C": 1.0, "gamma": "scale"},
    {"type": "rbf", "C": 5.0, "gamma": "scale"},
)


def load_config_overrides(path: str | None, default: Sequence[Dict]) -> List[Dict]:
    if path is None:
        return [cfg.copy() for cfg in default]
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Config override at {path} must be a list of dictionaries")
    normalized: List[Dict] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid config entry in {path}: {entry}")
        normalized.append(entry)
    return normalized


def run_random_forest(feature_set: FeatureSet, feature_dim: int, configs: Sequence[Dict]) -> List[Dict]:
    results = []
    for cfg in configs:
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            n_jobs=-1,
            random_state=42,
        )
        model.fit(feature_set.train_X, feature_set.train_y)
        train_time = time.time() - start
        metrics = evaluate_model(
            model,
            feature_set.val_X,
            feature_set.val_y,
            feature_set.test_X,
            feature_set.test_y,
        )
        run = {
            "model": "RandomForest",
            "params": cfg,
            "train_time_sec": train_time,
            "feature_dim": feature_dim,
        }
        run.update(metrics)
        results.append(run)
    return results


def run_svm(feature_set: FeatureSet, feature_dim: int, configs: Sequence[Dict]) -> List[Dict]:
    results = []
    for cfg in configs:
        if cfg["type"] == "linear":
            model = LinearSVC(C=cfg["C"], max_iter=5000)
        else:
            model = SVC(C=cfg["C"], kernel="rbf", gamma=cfg["gamma"])

        start = time.time()
        model.fit(feature_set.train_X, feature_set.train_y)
        train_time = time.time() - start
        metrics = evaluate_model(
            model,
            feature_set.val_X,
            feature_set.val_y,
            feature_set.test_X,
            feature_set.test_y,
        )
        run = {
            "model": "LinearSVM" if cfg["type"] == "linear" else "RBFSVM",
            "params": cfg,
            "train_time_sec": train_time,
            "feature_dim": feature_dim,
        }
        run.update(metrics)
        results.append(run)
    return results


def save_results(support_ratio: float, runs: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"classic_ml_support_{support_ratio:.2f}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    print(f"Saved classic ML results for support {support_ratio:.2f} to {output_path.relative_to(REPO_ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train classic ML models on mined motif features.")
    parser.add_argument(
        "--rf-config",
        type=str,
        help="Path to JSON list overriding the Random Forest hyperparameter grid.",
    )
    parser.add_argument(
        "--svm-config",
        type=str,
        help="Path to JSON list overriding the SVM hyperparameter grid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rf_configs = load_config_overrides(args.rf_config, DEFAULT_RF_CONFIGS)
    svm_configs = load_config_overrides(args.svm_config, DEFAULT_SVM_CONFIGS)
    support_ratios = sorted(
        float(p.name.split("_")[1]) for p in FEATURES_DIR.glob("support_*") if p.is_dir()
    )
    if not support_ratios:
        raise RuntimeError(f"No feature directories found in {FEATURES_DIR}")

    for support_ratio in support_ratios:
        print(f"\n=== Training classic models for support ratio {support_ratio:.2f} ===")
        feature_set = load_feature_set(support_ratio)
        feature_dim = feature_set.metadata.get("num_patterns", feature_set.train_X.shape[1])
        runs = []
        runs.extend(run_random_forest(feature_set, feature_dim, rf_configs))
        runs.extend(run_svm(feature_set, feature_dim, svm_configs))
        for run in runs:
            print(
                f"{run['model']} {run['params']} | "
                f"Val Acc {run['val_accuracy']:.3f} F1 {run['val_f1']:.3f} | "
                f"Test Acc {run['test_accuracy']:.3f} F1 {run['test_f1']:.3f} | "
                f"Train {run['train_time_sec']:.2f}s "
                f"ValInfer {run['val_inference_time_sec']:.3f}s TestInfer {run['test_inference_time_sec']:.3f}s"
            )
        save_results(support_ratio, runs)


if __name__ == "__main__":
    main()
