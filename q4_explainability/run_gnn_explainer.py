#!/usr/bin/env python3
"""Rebuild the GNN explainability runner with richer bookkeeping."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import load_mutag  # noqa: E402
from q2_gnn.models import GCN, GIN  # noqa: E402
from q2_gnn.training import evaluate, train_epoch  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGGREGATE_CSV = OUTPUT_DIR / "gnn_explainer_metrics.csv"
DEFAULT_CONFIGS: Dict[str, Dict[str, float]] = {
    "GCN": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.2, "lr": 0.005},
    "GIN": {"hidden_dim": 64, "num_layers": 3, "dropout": 0.2, "lr": 0.005},
}


@dataclass
class ExplanationRow:
    model: str
    seed: int
    split: str
    edge_keep_ratio: float
    graph_idx: int
    predicted_class: int
    original_prob: float
    fidelity_pos: float
    fidelity_neg: float
    sparsity: float
    edge_fraction_kept: float
    kept_edges: int
    num_edges: int
    runtime_sec: float


@dataclass
class ExplainConfig:
    models: Sequence[str]
    seeds: Sequence[int]
    keep_ratios: Sequence[float]
    max_graphs: int
    train_epochs: int
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNNExplainer across seeds/ratios.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_CONFIGS.keys()), help="Models to run (GCN, GIN).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds / splits (default 0 1 2).")
    parser.add_argument(
        "--edge-keep-ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Fractions of edges to keep when computing fidelity⁺ (default: 0.3 0.5 0.7 0.9).",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=-1,
        help="Optionally cap the number of correctly classified graphs explained per split (default -1 = all).",
    )
    parser.add_argument("--train-epochs", type=int, default=50, help="Training epochs for each run (default 50).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for MUTAG dataloaders (default 32).")
    return parser.parse_args()


def build_model(name: str, config: Dict[str, float], num_features: int, num_classes: int) -> torch.nn.Module:
    model_cls = GCN if name.upper() == "GCN" else GIN
    model = model_cls(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    return model.to(DEVICE)


def train_model(model: torch.nn.Module, loaders, lr: float, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_epoch(model, optimizer, loaders.train)
    evaluate(model, loaders.val)


def collect_correct_graphs(
    model: torch.nn.Module,
    loaders,
    limit: int,
) -> List[Tuple[Data, str, int]]:
    collected: List[Tuple[Data, str, int]] = []
    model.eval()
    for split_name in ("val", "test"):
        loader = getattr(loaders, split_name)
        running_idx = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                logits = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
            graphs = batch.to_data_list()
            for local_idx, graph in enumerate(graphs):
                absolute_idx = running_idx + local_idx
                if preds[local_idx].item() != graph.y.item():
                    continue
                graph = graph.cpu()
                graph.pred = preds[local_idx].cpu()
                graph.probs = probs[local_idx].cpu()
                collected.append((graph, split_name, absolute_idx))
                if limit >= 0 and len(collected) >= limit:
                    return collected
            running_idx += len(graphs)
    return collected


def build_masks(edge_mask: torch.Tensor, keep_ratio: float) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    num_edges = edge_mask.numel()
    if num_edges < 2:
        return None, None
    keep_edges = int(round(num_edges * keep_ratio))
    keep_edges = min(max(1, keep_edges), num_edges - 1)
    sorted_idx = torch.argsort(edge_mask)
    keep_idx = sorted_idx[-keep_edges:]
    keep_mask = torch.zeros(num_edges, dtype=torch.bool)
    keep_mask[keep_idx] = True
    drop_mask = ~keep_mask
    if drop_mask.sum().item() == 0:
        return None, None
    return keep_mask, drop_mask


def apply_mask(graph: Data, mask: torch.Tensor) -> Data | None:
    if mask.sum().item() == 0:
        return None
    subgraph = Data(x=graph.x.clone(), edge_index=graph.edge_index[:, mask], y=graph.y)
    # Preserve prediction metadata needed for evaluation.
    subgraph.pred = graph.pred
    subgraph.probs = graph.probs
    return subgraph


def evaluate_subgraph(model: torch.nn.Module, graph: Data) -> float:
    data = graph.to(DEVICE)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, batch)
        probs = F.softmax(logits, dim=-1)
    return probs.cpu()[0, graph.pred].item()


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
    graphs: Iterable[Tuple[Data, str, int]],
    keep_ratio: float,
    seed: int,
) -> List[ExplanationRow]:
    explainer = make_explainer(model)
    rows: List[ExplanationRow] = []
    for graph, split_name, graph_idx in graphs:
        data = graph.to(DEVICE)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)

        start = time.time()
        explanation = explainer(x=data.x, edge_index=data.edge_index, batch=batch)
        runtime = time.time() - start

        edge_mask = explanation.edge_mask.detach().cpu()
        keep_mask, drop_mask = build_masks(edge_mask, keep_ratio)
        if keep_mask is None or drop_mask is None:
            continue

        kept_graph = apply_mask(graph, keep_mask)
        dropped_graph = apply_mask(graph, drop_mask)
        if kept_graph is None or dropped_graph is None:
            continue

        original_prob = graph.probs[graph.pred].item()
        kept_prob = evaluate_subgraph(model, kept_graph)
        dropped_prob = evaluate_subgraph(model, dropped_graph)
        edge_fraction_kept = keep_mask.sum().item() / edge_mask.numel()

        rows.append(
            ExplanationRow(
                model=model_name,
                seed=seed,
                split=split_name,
                edge_keep_ratio=keep_ratio,
                graph_idx=graph_idx,
                predicted_class=int(graph.pred.item()),
                original_prob=original_prob,
                fidelity_pos=kept_prob,
                fidelity_neg=max(0.0, original_prob - dropped_prob),
                sparsity=1.0 - edge_fraction_kept,
                edge_fraction_kept=edge_fraction_kept,
                kept_edges=int(keep_mask.sum().item()),
                num_edges=int(edge_mask.numel()),
                runtime_sec=runtime,
            )
        )
    return rows


def run_for_model(
    model_name: str,
    config: Dict[str, float],
    seed: int,
    keep_ratios: Sequence[float],
    max_graphs: int,
    train_epochs: int,
    batch_size: int,
) -> List[ExplanationRow]:
    dataset, _, loaders = load_mutag(batch_size=batch_size, shuffle=True, seed=seed)
    model = build_model(model_name, config, dataset.num_features, dataset.num_classes)
    print(f"Training {model_name} (seed={seed})...", flush=True)
    train_model(model, loaders, config["lr"], train_epochs)

    graphs = collect_correct_graphs(model, loaders, max_graphs)
    if not graphs:
        print(f"No correctly classified graphs for {model_name} seed {seed}; skipping.", flush=True)
        return []

    print(f"Explaining {len(graphs)} graphs for {model_name} (seed={seed})...", flush=True)
    rows: List[ExplanationRow] = []
    for ratio in keep_ratios:
        rows.extend(explain_graphs(model_name, model, graphs, ratio, seed))

    if rows:
        model_dir = OUTPUT_DIR / model_name / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([asdict(row) for row in rows])
        df.to_csv(model_dir / "gnn_explainer_metrics.csv", index=False)
        print(f"Saved {model_name} seed {seed} metrics -> {model_dir}/gnn_explainer_metrics.csv ({len(df)} rows).", flush=True)
    return rows


def run_experiments(cfg: ExplainConfig) -> List[ExplanationRow]:
    all_rows: List[ExplanationRow] = []
    for seed in cfg.seeds:
        for model_name in cfg.models:
            if model_name not in DEFAULT_CONFIGS:
                print(f"Skipping unknown model '{model_name}'.", flush=True)
                continue
            all_rows.extend(
                run_for_model(
                    model_name,
                    DEFAULT_CONFIGS[model_name],
                    seed,
                    cfg.keep_ratios,
                    cfg.max_graphs,
                    cfg.train_epochs,
                    cfg.batch_size,
                )
            )
    return all_rows


def main() -> None:
    args = parse_args()
    cfg = ExplainConfig(
        models=[m.upper() for m in args.models],
        seeds=args.seeds,
        keep_ratios=args.edge_keep_ratios,
        max_graphs=args.max_graphs,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
    )
    rows = run_experiments(cfg)
    if not rows:
        print("No explanation rows produced.")
        return
    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(AGGREGATE_CSV, index=False)
    print(f"Saved aggregated explanation metrics -> {AGGREGATE_CSV} ({len(df)} rows).")


if __name__ == "__main__":
    main()
