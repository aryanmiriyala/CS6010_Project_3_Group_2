#!/usr/bin/env python3
"""Generate Q4 explainability plots for GNNs and classical models."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
GNNS_FIG_DIR = FIGURES_DIR / "gnn"
CLASSIC_FIG_DIR = FIGURES_DIR / "classic"
GNNS_FIG_DIR.mkdir(parents=True, exist_ok=True)
CLASSIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = FIGURES_DIR / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "DejaVu Sans"


def plot_gnn_metrics() -> None:
    df = pd.read_csv(RESULTS_DIR / "gnn_explainer_metrics.csv")
    seeds = sorted(df["seed"].unique())
    metrics = ["fidelity_pos", "fidelity_neg", "sparsity", "runtime_sec"]

    # Per-seed summaries
    for metric in metrics:
        plt.figure(figsize=(6, 4))
        for model in df["model"].unique():
            subset = df[df["model"] == model]
            means = subset.groupby("seed")[metric].mean()
            plt.plot(
                means.index,
                means.values,
                marker="o",
                label=model,
            )
        plt.title(f"{metric.replace('_', ' ').title()} per Seed")
        plt.xlabel("Seed")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(seeds)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(GNNS_FIG_DIR / f"{metric}_per_seed.png", dpi=200)
        plt.close()

    # Bar plot: fidelity_pos vs fidelity_neg per model
    models = df["model"].unique()
    fidelity_metrics = []
    for metric in ("fidelity_pos", "fidelity_neg"):
        for model in models:
            subset = df[df["model"] == model][metric]
            fidelity_metrics.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": subset.mean(),
                    "std": subset.std(),
                }
            )
    fidelity_df = pd.DataFrame(fidelity_metrics)

    plt.figure(figsize=(6, 4))
    width = 0.35
    x = range(len(models))
    for i, metric in enumerate(("fidelity_pos", "fidelity_neg")):
        metric_df = fidelity_df[fidelity_df["metric"] == metric]
        offsets = [val + (i - 0.5) * width for val in x]
        plt.bar(
            offsets,
            metric_df["mean"],
            width=width,
            label=metric.replace("_", " "),
            yerr=metric_df["std"],
            capsize=4,
        )
    plt.xticks(x, models)
    plt.title("GNNExplainer Fidelity Metrics")
    plt.ylabel("Value")
    plt.ylim(0, 1.05)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(GNNS_FIG_DIR / "fidelity_bar.png", dpi=200)
    plt.close()

    # Scatter: fidelity_pos vs fidelity_neg colored by model
    # Sufficiency vs necessity bar chart per seed
    fig, axes = plt.subplots(len(models), 1, figsize=(6, 3 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        subset = df[df["model"] == model]
        sufficiency = subset.groupby("seed")["fidelity_pos"].mean()
        necessity = subset.groupby("seed")["fidelity_neg"].mean()
        x = range(len(sufficiency))
        width = 0.35
        ax.bar([val - width / 2 for val in x], sufficiency, width=width, label="Fidelity⁺ (keep)", color="#1f77b4")
        ax.bar([val + width / 2 for val in x], necessity, width=width, label="Fidelity⁻ (drop)", color="#ff7f0e")
        ax.set_title(f"{model}: Sufficiency vs Necessity per Seed")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(sufficiency.index)
    axes[-1].set_xlabel("Seed")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    fig.savefig(GNNS_FIG_DIR / "fidelity_sufficiency_necessity.png", dpi=200)
    plt.close(fig)

    # Runtime bar plot
    plt.figure(figsize=(5, 4))
    means = df.groupby("model")["runtime_sec"].mean()
    stds = df.groupby("model")["runtime_sec"].std()
    plt.bar(means.index, means.values, yerr=stds.values, capsize=4)
    plt.title("GNNExplainer Runtime per Graph")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(GNNS_FIG_DIR / "runtime_bar.png", dpi=200)
    plt.close()


def plot_classic_metrics(top_k: int = 10) -> None:
    df = pd.read_csv(RESULTS_DIR / "classic_motif_importances.csv")

    # Mean importance vs support ratio
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

    # Per-seed importance trends
    seed_agg = (
        df.groupby(["seed", "model", "support_ratio"])["importance"]
        .mean()
        .reset_index()
        .sort_values("support_ratio")
    )
    seeds = sorted(seed_agg["seed"].unique())
    num_seeds = len(seeds)
    fig, axes = plt.subplots(1, num_seeds, figsize=(4 * num_seeds, 3), sharey=True)
    if num_seeds == 1:
        axes = [axes]
    colors = {"LinearSVM": "#1f77b4", "RandomForest": "#ff7f0e"}
    for ax, seed in zip(axes, seeds):
        subset = seed_agg[seed_agg["seed"] == seed]
        for model in subset["model"].unique():
            model_slice = subset[subset["model"] == model]
            ax.plot(
                model_slice["support_ratio"],
                model_slice["importance"],
                marker="o",
                label=model,
                color=colors.get(model),
            )
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Support ratio")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean importance")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CLASSIC_FIG_DIR / "per_seed_importance_vs_support.png", dpi=200)
    plt.close(fig)

    # Class-label distribution among top-K motifs per model
    class_counts = (
        df.groupby(["model", "class_label"])
        .size()
        .reset_index(name="count")
    )
    plt.figure(figsize=(5, 4))
    models = class_counts["model"].unique()
    x = range(len(models))
    width = 0.35
    for i, class_label in enumerate(sorted(class_counts["class_label"].unique())):
        subset = class_counts[class_counts["class_label"] == class_label]
        offsets = [val + (i - 0.5) * width for val in x]
        plt.bar(
            offsets,
            subset["count"],
            width=width,
            label=f"class {class_label}",
        )
    plt.xticks(x, models)
    plt.title(f"Class Labels among Top-{top_k} Motifs")
    plt.ylabel("Count")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(CLASSIC_FIG_DIR / "class_label_distribution.png", dpi=200)
    plt.close()

    # Class-label distribution per seed
    seed_class_counts = (
        df.groupby(["seed", "model", "class_label"])
        .size()
        .reset_index(name="count")
    )
    fig, axes = plt.subplots(1, num_seeds, figsize=(4 * num_seeds, 3), sharey=True)
    if num_seeds == 1:
        axes = [axes]
    width = 0.35
    for ax, seed in zip(axes, seeds):
        subset = seed_class_counts[seed_class_counts["seed"] == seed]
        models = subset["model"].unique()
        x = range(len(models))
        for i, class_label in enumerate(sorted(subset["class_label"].unique())):
            class_slice = subset[subset["class_label"] == class_label]
            offsets = [val + (i - 0.5) * width for val in x]
            ax.bar(
                offsets,
                class_slice["count"],
                width=width,
                label=f"class {class_label}" if seed == seeds[0] else "",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20)
        ax.set_title(f"Seed {seed}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(CLASSIC_FIG_DIR / "per_seed_class_label_distribution.png", dpi=200)
    plt.close(fig)

    # Classical vs GNN comparative scatter (importance vs support vs fidelity)
    try:
        gnn_df = pd.read_csv(RESULTS_DIR / "gnn_explainer_metrics.csv")
        comp_df = df.groupby(["seed", "model"])["importance"].mean().reset_index()
        gnn_summary = (
            gnn_df.groupby(["seed", "model"])[["fidelity_pos", "fidelity_neg", "sparsity"]]
            .mean()
            .reset_index()
        )
        merge_df = (
            comp_df.merge(gnn_summary, on="seed", how="outer", suffixes=("_classic", "_gnn"))
            .dropna(subset=["importance", "fidelity_pos"])
        )

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
