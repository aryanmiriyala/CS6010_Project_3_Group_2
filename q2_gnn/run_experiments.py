import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Add repo root to path so we can import data_access
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import load_mutag
from q2_gnn.models import GCN, GIN
from q2_gnn.training import evaluate, train_epoch

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configuration (edit values here instead of using CLI arguments).
EXPERIMENT_SEEDS = [0, 1, 2]
NUM_EPOCHS = 50
BATCH_SIZE = 32
RUN_GCN = True
RUN_GIN = True

HIDDEN_DIMS = [32, 64]
NUM_LAYERS = [2, 3, 4]
DROPOUTS = [0.0, 0.2, 0.75]
LRS = [0.001, 0.005]


def run_model_experiments(model_name: str, ModelClass) -> None:
    """Train/evaluate a model class across the hyperparameter grid for each seed."""

    for seed in EXPERIMENT_SEEDS:
        print(f"\n=== {model_name} Experiments (seed={seed}) ===")
        dataset, _, loaders = load_mutag(batch_size=BATCH_SIZE, shuffle=True, seed=seed)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        seed_results: list[dict] = []

        for hidden_dim, layers, dropout, lr in itertools.product(
            HIDDEN_DIMS, NUM_LAYERS, DROPOUTS, LRS
        ):
            model = ModelClass(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=layers,
                dropout=dropout,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            train_start = time.time()
            for _ in range(NUM_EPOCHS):
                train_epoch(model, optimizer, loaders.train)
            train_runtime = time.time() - train_start

            val_metrics, val_inf_time = evaluate(model, loaders.val)
            test_metrics, test_inf_time = evaluate(model, loaders.test)

            seed_results.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "layers": layers,
                    "dropout": dropout,
                    "lr": lr,
                    "num_params": num_params,
                    "train_epochs": NUM_EPOCHS,
                    "train_time_sec": train_runtime,
                    "val_inference_time_sec": val_inf_time,
                    "test_inference_time_sec": test_inf_time,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "test_auc": test_metrics["auc"],
                }
            )

            print(
                f"{model_name} seed={seed} h={hidden_dim} L={layers} d={dropout} lr={lr} | "
                f"Val Acc {val_metrics['accuracy']:.3f} AUC {val_metrics['auc']:.3f} | "
                f"Test Acc {test_metrics['accuracy']:.3f} AUC {test_metrics['auc']:.3f} | "
                f"Train {train_runtime:.1f}s"
            )

        save_results(model_name, seed, seed_results)


def save_results(model_name: str, seed: int, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    output_path = seed_dir / f"{model_name.lower()}_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {model_name} results to {output_path.relative_to(REPO_ROOT)}")


def main() -> None:
    if RUN_GCN:
        run_model_experiments("GCN", GCN)
    if RUN_GIN:
        run_model_experiments("GIN", GIN)


if __name__ == "__main__":
    main()
