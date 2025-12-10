#!/usr/bin/env python3
"""Generate Q4 explainability plots for GNNs and classical models."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
GNNS_FIG_DIR = FIGURES_DIR / "gnn"
CLASSIC_FIG_DIR = FIGURES_DIR / "classic"
for directory in (FIGURES_DIR, GNNS_FIG_DIR, CLASSIC_FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
MPL_DIR = FIGURES_DIR / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "DejaVu Sans"


def _grouped_ratio_bars(ax, pivot: pd.DataFrame, ratios: list[float], colors: dict[float, str]):
    seeds = pivot.index.tolist()
    x = np.arange(len(seeds))
    width = 0.8 / max(1, len(ratios))
    bars = []
    labels = []
    for idx, ratio in enumerate(ratios):
        if ratio not in pivot.columns:
            continue
        offsets = x + (idx - (len(ratios) - 1) / 2) * width
        bar = ax.bar(
            offsets,
            pivot[ratio].values,
            width=width * 0.9,
            color=colors[ratio],
            label=f"keep={ratio:.2f}",
        )
        bars.append(bar)
        labels.append(f"keep={ratio:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.grid(True, axis="y", alpha=0.3)
    return bars, labels


def plot_gnn_metrics() -> None:
    df = pd.read_csv(RESULTS_DIR / "gnn_explainer_metrics.csv")
    models = sorted(df["model"].unique())
    ratios = sorted(df["edge_keep_ratio"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {ratio: cmap(i % cmap.N) for i, ratio in enumerate(ratios)}

    # Remove obsolete single-model figures so only combined ones remain.
    for pattern in ("GCN_*.png", "GIN_*.png"):
        for stale_path in GNNS_FIG_DIR.glob(pattern):
            try:
                stale_path.unlink()
            except OSError:
                pass

    metric_specs = [
        {
            "metrics": [("fidelity_pos", "Fidelity⁺"), ("fidelity_neg", "Fidelity⁻")],
            "filename": "gnn_fidelity_vs_seed_ratio",
        },
        {
            "metrics": [("runtime_sec", "Runtime (s)")],
            "filename": "gnn_runtime_vs_seed_ratio",
        },
        {
            "metrics": [("kept_edges", "Kept edges"), ("num_edges", "Total edges")],
            "filename": "gnn_edges_vs_seed_ratio",
        },
        {
            "metrics": [("sparsity", "Sparsity"), ("edge_fraction_kept", "Edge fraction kept")],
            "filename": "gnn_sparsity_vs_seed_ratio",
        },
    ]

    for spec in metric_specs:
        metrics = spec["metrics"]
        filename = spec["filename"]
        total_plots = len(metrics) * len(models)
        fig, axes = plt.subplots(
            1,
            total_plots,
            figsize=(4 * total_plots, 3),
            sharey=False,
        )
        axes = np.asarray(axes).ravel()

        legend_handles: list = []
        legend_labels: list[str] = []
        ax_idx = 0
        for metric, ylabel in metrics:
            for model in models:
                ax = axes[ax_idx]
                subset = df[df["model"] == model]
                seeds = sorted(subset["seed"].unique())
                pivot = (
                    subset.groupby(["seed", "edge_keep_ratio"])[metric]
                    .mean()
                    .unstack("edge_keep_ratio")
                    .reindex(seeds)
                )
                handles, labels = _grouped_ratio_bars(ax, pivot, ratios, colors)
                ax.set_title(f"{model} — {ylabel}")
                ax.set_xlabel("Seed")
                if ax_idx % len(models) == 0:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel("")
                if not legend_handles and handles:
                    legend_handles = handles
                    legend_labels = labels
                ax_idx += 1
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_handles)))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(GNNS_FIG_DIR / f"{filename}.png", dpi=200)
        plt.close(fig)


def plot_classic_metrics(top_k: int = 10) -> None:
    df = pd.read_csv(RESULTS_DIR / "classic_motif_importances.csv")

    agg = (
        df.groupby(["model", "support_ratio"])["importance"]
        .agg(["mean", "std"])
        .reset_index()
    )
    plt.figure(figsize=(6, 4))
    for model in agg["model"].unique():
        subset = agg[agg["model"] == model]
        plt.errorbar(
            subset["support_ratio"],
            subset["mean"],
            yerr=subset["std"],
            label=model,
            marker="o",
            capsize=4,
        )
    plt.title(f"Average Motif Importance (top-{top_k})")
    plt.ylabel("Mean importance / |coef|")
    plt.tight_layout()
    plt.savefig(CLASSIC_FIG_DIR / "importance_vs_support.png", dpi=200)
    plt.close()

    seed_agg = (
        df.groupby(["seed", "model", "support_ratio"])["importance"]
        .mean()
        .reset_index()
    )
    seeds = sorted(seed_agg["seed"].unique())
    support_vals = sorted(seed_agg["support_ratio"].unique())
    models = sorted(seed_agg["model"].unique())
    colors = {"LinearSVM": "#1f77b4", "RandomForest": "#ff7f0e", "RBFSVM": "#2ca02c"}
    fig, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    width = 0.8 / max(1, len(models))
    x = np.arange(len(support_vals))
    for ax, seed in zip(axes, seeds):
        subset = seed_agg[seed_agg["seed"] == seed]
        for idx, model in enumerate(models):
            model_slice = (
                subset[subset["model"] == model]
                .set_index("support_ratio")
                .reindex(support_vals)
            )
            offsets = x + (idx - (len(models) - 1) / 2) * width
            ax.bar(
                offsets,
                model_slice["importance"],
                width=width * 0.9,
                color=colors.get(model),
                label=model if seed == seeds[0] else "",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.2f}" for s in support_vals])
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Support ratio")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean importance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(CLASSIC_FIG_DIR / "per_seed_importance_vs_support.png", dpi=200)
    plt.close(fig)

    class_counts = (
        df.groupby(["model", "class_label"])
        .size()
        .reset_index(name="count")
    )
    plt.figure(figsize=(6, 4))
    models = class_counts["model"].unique()
    x = np.arange(len(models))
    width = 0.35
    for idx, class_label in enumerate(sorted(class_counts["class_label"].unique())):
        subset = class_counts[class_counts["class_label"] == class_label]
        offsets = x + (idx - 0.5) * width
        plt.bar(offsets, subset["count"], width=width, label=f"class {class_label}")
    plt.xticks(x, models)
    plt.title(f"Class Labels among Top-{top_k} Motifs")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLASSIC_FIG_DIR / "class_label_distribution.png", dpi=200)
    plt.close()

    avg_class_counts = (
        df.groupby(["seed", "model", "support_ratio", "class_label"])
        .size()
        .reset_index(name="count")
        .groupby(["model", "support_ratio", "class_label"])["count"]
        .mean()
        .reset_index()
    )
    class_labels = sorted(df["class_label"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    width = 0.8 / max(1, len(class_labels))
    x = np.arange(len(support_vals))
    for ax, model in zip(axes, models):
        subset = avg_class_counts[avg_class_counts["model"] == model]
        for idx, class_label in enumerate(class_labels):
            class_slice = (
                subset[subset["class_label"] == class_label]
                .set_index("support_ratio")
                .reindex(support_vals)
                .fillna(0)
            )
            offsets = x + (idx - (len(class_labels) - 1) / 2) * width
            ax.bar(
                offsets,
                class_slice["count"],
                width=width * 0.9,
                label=f"class {class_label}" if ax == axes[0] else "",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.2f}" for s in support_vals], rotation=20)
        ax.set_title(model)
        ax.set_xlabel("Support ratio")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Avg. motif count (top-20)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend([h for h in handles if h.get_label()], [l for l in labels if l], loc="upper center", ncol=len(class_labels), bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(CLASSIC_FIG_DIR / "per_seed_class_label_distribution.png", dpi=200)
    plt.close(fig)

    try:
        gnn_df = pd.read_csv(RESULTS_DIR / "gnn_explainer_metrics.csv")
        comp_df = df.groupby(["seed", "model"])["importance"].mean().reset_index()
        gnn_summary = (
            gnn_df.groupby(["seed", "model"])[["fidelity_pos", "fidelity_neg", "sparsity"]]
            .mean()
            .reset_index()
        )
        merge_df = comp_df.merge(gnn_summary, on="seed", suffixes=("_classic", "_gnn"))
        if not merge_df.empty:
            plt.figure(figsize=(6, 4))
            for seed in merge_df["seed"].unique():
                subset = merge_df[merge_df["seed"] == seed]
                plt.scatter(
                    subset["importance"],
                    subset["fidelity_pos"],
                    s=50,
                    label=f"Seed {seed}",
                )
            plt.xlabel("Mean Classical Motif Importance")
            plt.ylabel("Mean GNN Fidelity⁺")
            plt.title("Classical vs GNN Explainability (per seed)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(CLASSIC_FIG_DIR / "classic_vs_gnn_comparison.png", dpi=200)
            plt.close()
    except Exception:
        pass


def main() -> None:
    plot_gnn_metrics()
    plot_classic_metrics()
    print(f"Saved plots under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
