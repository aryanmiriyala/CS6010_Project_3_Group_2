#!/usr/bin/env python3
"""Aggregate Q1 vs Q2 metrics and emit per-support/per-seed comparison plots."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
Q1_ROOT = REPO_ROOT / "q1_frequent_subgraphs_classic_ml"
Q2_ROOT = REPO_ROOT / "q2_gnn"
RESULTS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = RESULTS_DIR / "figures"
MPL_DIR = RESULTS_DIR / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
QUALITY_DIR = FIGURES_DIR / "quality"
EFFICIENCY_DIR = FIGURES_DIR / "efficiency"
QUALITY_DIR.mkdir(parents=True, exist_ok=True)
EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

QUALITY_METRICS = ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc")
EFFICIENCY_COMPONENTS = ("preprocess_time_sec", "train_time_sec", "test_inference_time_sec")


def _parse_seed(name: str) -> int | None:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return None


def _parse_support(name: str) -> float | None:
    try:
        if name.startswith("support_"):
            name = name.split("_", 1)[1]
        return float(name)
    except (IndexError, ValueError):
        return None


def collect_q1_preprocessing() -> Dict[Tuple[int, float], float]:
    artifacts_dir = Q1_ROOT / "artifacts"
    features_dir = Q1_ROOT / "features"
    mapping: Dict[Tuple[int, float], float] = {}

    for feature_seed_dir in sorted(features_dir.glob("seed_*")):
        seed = _parse_seed(feature_seed_dir.name)
        if seed is None:
            continue
        for support_dir in sorted(feature_seed_dir.glob("support_*")):
            support = _parse_support(support_dir.name)
            if support is None:
                continue
            feature_time = 0.0
            feature_config_path = support_dir / "feature_config.json"
            if feature_config_path.exists():
                metadata = json.loads(feature_config_path.read_text())
                feature_time = float(metadata.get("feature_construction_runtime_sec", 0.0))

            mining_time = 0.0
            artifact_seed_dir = artifacts_dir / f"seed_{seed}"
            for class_dir in sorted(artifact_seed_dir.glob("class_*")):
                patterns_path = class_dir / f"support_{support:.2f}" / "patterns.json"
                if patterns_path.exists():
                    payload = json.loads(patterns_path.read_text())
                    mining_time += float(payload.get("runtime_sec", 0.0))

            mapping[(seed, round(support, 2))] = feature_time + mining_time
    return mapping


def load_q1_results(preprocess_map: Dict[Tuple[int, float], float]) -> List[dict]:
    rows: List[dict] = []
    results_dir = Q1_ROOT / "results"
    for seed_dir in sorted(results_dir.glob("seed_*")):
        seed = _parse_seed(seed_dir.name)
        if seed is None:
            continue
        for csv_path in sorted(seed_dir.glob("classic_ml_support_*.csv")):
            support_ratio = _parse_support(csv_path.stem.split("support_")[-1])
            if support_ratio is None:
                continue
            preprocess_time = preprocess_map.get((seed, round(support_ratio, 2)), 0.0)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "family": "Classic",
                        "approach": row["model"],
                        "support_ratio": support_ratio,
                        "seed": seed,
                        "config_label": f"{row['model']} support={support_ratio:.2f}",
                        "train_time_sec": row.get("train_time_sec", 0.0),
                        "val_inference_time_sec": row.get("val_inference_time_sec", 0.0),
                        "test_inference_time_sec": row.get("test_inference_time_sec", 0.0),
                        "val_accuracy": row.get("val_accuracy"),
                        "val_precision": row.get("val_precision"),
                        "val_recall": row.get("val_recall"),
                        "val_f1": row.get("val_f1"),
                        "val_auc": row.get("val_auc"),
                        "test_accuracy": row.get("test_accuracy"),
                        "test_precision": row.get("test_precision"),
                        "test_recall": row.get("test_recall"),
                        "test_f1": row.get("test_f1"),
                        "test_auc": row.get("test_auc"),
                        "preprocess_time_sec": preprocess_time,
                    }
                )
    return rows


def load_q2_results() -> List[dict]:
    rows: List[dict] = []
    results_dir = Q2_ROOT / "results"
    for seed_dir in sorted(results_dir.glob("seed_*")):
        seed = _parse_seed(seed_dir.name)
        if seed is None:
            continue
        for model_name in ("gcn", "gin"):
            csv_path = seed_dir / f"{model_name}_results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "family": model_name.upper(),
                        "approach": model_name.upper(),
                        "support_ratio": None,
                        "seed": seed,
                        "config_label": f"{model_name.upper()} h={row['hidden_dim']} L={row['layers']} d={row['dropout']} lr={row['lr']}",
                        "train_time_sec": row.get("train_time_sec", 0.0),
                        "val_inference_time_sec": row.get("val_inference_time_sec", 0.0),
                        "test_inference_time_sec": row.get("test_inference_time_sec", 0.0),
                        "val_accuracy": row.get("val_accuracy"),
                        "val_precision": row.get("val_precision"),
                        "val_recall": row.get("val_recall"),
                        "val_f1": row.get("val_f1"),
                        "val_auc": row.get("val_auc"),
                        "test_accuracy": row.get("test_accuracy"),
                        "test_precision": row.get("test_precision"),
                        "test_recall": row.get("test_recall"),
                        "test_f1": row.get("test_f1"),
                        "test_auc": row.get("test_auc"),
                        "preprocess_time_sec": 0.0,
                    }
                )
    return rows


def aggregate() -> pd.DataFrame:
    preprocess_map = collect_q1_preprocessing()
    data = load_q1_results(preprocess_map) + load_q2_results()
    if not data:
        raise SystemExit("No Q1/Q2 result files found. Run the pipelines first.")
    df = pd.DataFrame(data)
    df["total_pipeline_time_sec"] = df["preprocess_time_sec"].fillna(0.0) + df["train_time_sec"].fillna(0.0)
    return df


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "train_time_sec",
        "val_inference_time_sec",
        "test_inference_time_sec",
        "preprocess_time_sec",
        "total_pipeline_time_sec",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
    ]
    summary = df.groupby(["family", "approach", "config_label"]).agg({m: ["mean", "std"] for m in metrics}).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    best_rows = summary.sort_values(by="val_accuracy_mean", ascending=False).groupby("approach").head(1)
    model_summary = df.groupby("approach")[metrics].mean().reset_index()
    family_summary = df.groupby("family")[metrics].mean().reset_index()
    return summary, best_rows, model_summary, family_summary


def _prepare_group_positions(num_groups: int, num_series: int, bar_width: float = 0.2):
    x = np.arange(num_groups)
    offsets = (np.arange(num_series) - (num_series - 1) / 2) * bar_width
    return x, offsets, bar_width


def plot_classical_support_metric(df: pd.DataFrame, metric: str) -> None:
    classic_df = df[df["family"] == "Classic"].dropna(subset=["support_ratio", metric])
    if classic_df.empty:
        return
    supports = sorted(classic_df["support_ratio"].unique())
    approaches = sorted(classic_df["approach"].unique())
    fig_width = max(8, len(supports) * len(approaches) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    bar_width = 0.8 / len(approaches)
    x = np.arange(len(supports))

    for idx, approach in enumerate(approaches):
        subset = (
            classic_df[classic_df["approach"] == approach]
            .groupby("support_ratio")[metric]
            .mean()
        )
        heights = [subset.get(support, np.nan) for support in supports]
        ax.bar(
            x + idx * bar_width,
            heights,
            width=bar_width,
            label=approach,
        )

    ax.set_xticks(x + (len(approaches) - 1) * bar_width / 2)
    ax.set_xticklabels([f"{s:.2f}" for s in supports])
    ax.set_xlabel("Support Ratio")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Classical {metric.replace('_', ' ').title()} (bar view)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(QUALITY_DIR / f"classic_support_{metric}.png", dpi=200)
    plt.close(fig)


def select_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.dropna(subset=["val_accuracy"]).copy()
    if subset.empty:
        return subset
    idx = subset.groupby(["approach", "seed"])["val_accuracy"].idxmax()
    best_df = subset.loc[idx].copy()
    return best_df


def plot_quality_summary(best_df: pd.DataFrame, metric: str) -> None:
    metric_df = best_df.dropna(subset=[metric])
    if metric_df.empty:
        return
    approaches = sorted(metric_df["approach"].unique())
    seeds = sorted(metric_df["seed"].unique())
    bar_width = 0.15
    x, offsets, bar_width = _prepare_group_positions(len(approaches), len(seeds), bar_width)
    fig_width = max(8, len(approaches) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    for idx, seed in enumerate(seeds):
        subset = metric_df[metric_df["seed"] == seed].set_index("approach")
        heights = [subset.loc[app, metric] if app in subset.index else np.nan for app in approaches]
        ax.bar(
            x + offsets[idx],
            heights,
            width=bar_width,
            label=f"Seed {seed}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} (best config per seed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(QUALITY_DIR / f"best_{metric}.png", dpi=200)
    plt.close(fig)


def plot_efficiency_components(best_df: pd.DataFrame) -> None:
    if best_df.empty:
        return
    approaches = sorted(best_df["approach"].unique())
    seeds = sorted(best_df["seed"].unique())
    metrics = ["preprocess_time_sec", "train_time_sec", "test_inference_time_sec", "total_pipeline_time_sec"]
    for metric in metrics:
        metric_df = best_df.dropna(subset=[metric])
        if metric_df.empty:
            continue
        x, offsets, bar_width = _prepare_group_positions(len(approaches), len(seeds))
        fig_width = max(8, len(approaches) * 0.9)
        fig, ax = plt.subplots(figsize=(fig_width, 4.5))
        for idx, seed in enumerate(seeds):
            subset = metric_df[metric_df["seed"] == seed].set_index("approach")
            heights = [subset.loc[app, metric] if app in subset.index else np.nan for app in approaches]
            ax.bar(
                x + offsets[idx],
                heights,
                width=bar_width,
                label=f"Seed {seed}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=20)
        ax.set_ylabel(metric.replace("_", " ").title() + " (s)")
        ax.set_title(f"{metric.replace('_', ' ').title()} per Approach (best configs)")
        ax.legend()
        fig.tight_layout()
        filename = metric.replace("_sec", "").replace("_", "") + "_bar.png"
        fig.savefig(EFFICIENCY_DIR / filename, dpi=200)
        plt.close(fig)


def main() -> None:
    df = aggregate()
    df.to_csv(RESULTS_DIR / "aggregated_results.csv", index=False)
    summary, best_rows, model_summary, family_summary = summarize(df)
    summary.to_csv(RESULTS_DIR / "summary_by_config.csv", index=False)
    best_rows.to_csv(RESULTS_DIR / "best_configs.csv", index=False)
    model_summary.to_csv(RESULTS_DIR / "summary_by_model.csv", index=False)
    family_summary.to_csv(RESULTS_DIR / "summary_by_family.csv", index=False)

    best_df = select_best_configs(df)

    for metric in QUALITY_METRICS:
        plot_classical_support_metric(df, metric)
        plot_quality_summary(best_df, metric)

    plot_efficiency_components(best_df)
    # Total time bars already included in efficiency components loop

    print("Saved aggregated comparison outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
