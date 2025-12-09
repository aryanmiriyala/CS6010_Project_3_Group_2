#!/usr/bin/env python3
"""Run GNNExplainer on the best-performing GNN models for MUTAG."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import load_mutag  # noqa: E402
from q2_gnn.models import GCN, GIN  # noqa: E402
from q2_gnn.training import evaluate, train_epoch  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TRAIN_EPOCHS = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGGREGATE_CSV = OUTPUT_DIR / "gnn_explainer_metrics.csv"

BEST_CONFIGS: Dict[str, Dict[str, float]] = {
    "GCN": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.2, "lr": 0.005},
    "GIN": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.2, "lr": 0.005},
}


def build_model(name: str, config: Dict[str, float], num_features: int, num_classes: int) -> torch.nn.Module:
    model_cls = GCN if name == "GCN" else GIN
    model = model_cls(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    return model.to(DEVICE)


def train_model(model: torch.nn.Module, train_loader, val_loader, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(TRAIN_EPOCHS):
        train_epoch(model, optimizer, train_loader)
    evaluate(model, val_loader)


def collect_correct_graphs(model: torch.nn.Module, loaders, limit: int) -> List[Tuple[Data, str]]:
    """Grab correctly classified graphs so explanations are faithful."""

    collected: List[Tuple[Data, str]] = []
    model.eval()
    for split_name in ("val", "test"):
        loader = getattr(loaders, split_name)
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                logits = model(batch.x, batch.edge_index, batch.batch)
                preds = logits.argmax(dim=-1)
                probs = F.softmax(logits, dim=-1)
            graphs = batch.to_data_list()
            for idx, graph in enumerate(graphs):
                if preds[idx].item() != graph.y.item():
                    continue
                graph = graph.cpu()
                graph.pred = preds[idx].cpu()
                graph.probs = probs[idx].cpu()
                collected.append((graph, split_name))
                if limit >= 0 and len(collected) >= limit:
                    return collected
    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNNExplainer for best GNN configs.")
    parser.add_argument(
        "--edge-keep-ratios",
        type=float,
        nargs="+",
        default=[0.3],
        help="Fractions of edges to keep when computing fidelity⁺ (default: 0.3).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Dataset split seeds to use for training/explaining (default: 0 1 2).",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=5,
        help=(
            "Maximum number of correctly classified test graphs to explain per model "
            "(set to -1 to explain all). Default: 5."
        ),
    )
    return parser.parse_args()


def build_masks(edge_mask: torch.Tensor, keep_ratio: float) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return boolean masks for (i) edges to keep, (ii) edges to drop."""

    num_edges = edge_mask.numel()
    if num_edges < 2:
        return None, None
    k = max(1, int(round(num_edges * keep_ratio)))
    if k >= num_edges:
        k = num_edges - 1
    if k <= 0:
        return None, None

    sorted_idx = torch.argsort(edge_mask)
    keep_idx = sorted_idx[-k:]

    keep_mask = torch.zeros(num_edges, dtype=torch.bool)
    keep_mask[keep_idx] = True
    drop_mask = ~keep_mask
    if drop_mask.sum().item() == 0:
        return None, None
    return keep_mask, drop_mask


def apply_edge_mask(graph: Data, mask: torch.Tensor) -> Data | None:
    if mask.sum().item() == 0:
        return None
    subgraph = Data(x=graph.x.clone(), edge_index=graph.edge_index[:, mask], y=graph.y)
    return subgraph


def evaluate_graph(model: torch.nn.Module, graph: Data) -> torch.Tensor:
    data = graph.to(DEVICE)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, batch)
        probs = F.softmax(logits, dim=-1)
    return probs.cpu()


def make_explainer(model: torch.nn.Module) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="log_probs",
        ),
        node_mask_type="attributes",
        edge_mask_type="object",
    )


def explain_graphs(
    model_name: str,
    model: torch.nn.Module,
    graphs: List[Tuple[Data, str]],
    keep_ratio: float,
    seed: int,
) -> List[Dict[str, float]]:
    explainer = make_explainer(model)
    rows: List[Dict[str, float]] = []

    for idx, (graph, split_name) in enumerate(graphs):
        data = graph.to(DEVICE)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)

        start = time.time()
        explanation = explainer(x=data.x, edge_index=data.edge_index, batch=batch)
        runtime = time.time() - start

        edge_mask = explanation.edge_mask.detach().cpu()
        keep_mask, drop_mask = build_masks(edge_mask, keep_ratio)
        if keep_mask is None or drop_mask is None:
            continue

        kept_graph = apply_edge_mask(graph, keep_mask)
        dropped_graph = apply_edge_mask(graph, drop_mask)
        if kept_graph is None or dropped_graph is None:
            continue

        kept_prob = evaluate_graph(model, kept_graph)[0, graph.pred].item()
        dropped_prob = evaluate_graph(model, dropped_graph)[0, graph.pred].item()
        orig_prob = graph.probs[graph.pred].item()

        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "split": split_name,
                "graph_idx": idx,
                "predicted_class": int(graph.pred.item()),
                "original_prob": orig_prob,
                "fidelity_pos": kept_prob,
                "fidelity_neg": max(0.0, orig_prob - dropped_prob),
                "sparsity": 1.0 - (keep_mask.sum().item() / edge_mask.numel()),
                "edge_fraction_kept": keep_mask.sum().item() / edge_mask.numel(),
                "num_edges": edge_mask.numel(),
                "edge_keep_ratio": keep_ratio,
                "runtime_sec": runtime,
            }
        )
    return rows


def run_for_model(
    model_name: str,
    config: Dict[str, float],
    keep_ratios: list[float],
    seed: int,
    max_graphs: int,
) -> List[Dict[str, float]]:
    dataset, _, loaders = load_mutag(batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    model = build_model(model_name, config, dataset.num_features, dataset.num_classes)
    print(f"Training {model_name} (seed={seed})...", flush=True)
    train_model(model, loaders.train, loaders.val, config["lr"])

    graphs = collect_correct_graphs(model, loaders, max_graphs)
    if not graphs:
        print(f"Skipping {model_name} (seed={seed}); no correctly classified graphs.", flush=True)
        return []

    print(f"Explaining {len(graphs)} graphs for {model_name} (seed={seed})...", flush=True)
    rows: List[Dict[str, float]] = []
    for ratio in keep_ratios:
        rows.extend(explain_graphs(model_name, model, graphs, ratio, seed))

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        seed_dir = model_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        csv_path = seed_dir / "gnn_explainer_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(
            f"Saved {model_name} explanation metrics to {csv_path} ({len(df)} rows).",
            flush=True,
        )
    return rows


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    all_rows: List[Dict[str, float]] = []
    for seed in args.seeds:
        for model_name, config in BEST_CONFIGS.items():
            all_rows.extend(
                run_for_model(
                    model_name,
                    config,
                    args.edge_keep_ratios,
                    seed,
                    args.max_graphs,
                )
            )

    if not all_rows:
        print("No explanation metrics were produced.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(AGGREGATE_CSV, index=False)
    print(f"Saved aggregated explanation metrics to {AGGREGATE_CSV} ({len(df)} rows).")


if __name__ == "__main__":
    main()
