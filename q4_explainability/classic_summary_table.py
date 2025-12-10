#!/usr/bin/env python3
"""Summarize classical explainability metrics into a table."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K_PER_CLASS = 5


def save_seed_level_summary(df: pd.DataFrame) -> None:
    summaries = df.groupby(["seed", "model", "support_ratio"])  # type: ignore[arg-type]
    table = (
        summaries["importance"].mean().reset_index(name="mean_importance")
        .merge(
            summaries.size()
            .rename("count")
            .reset_index()[["seed", "model", "support_ratio", "count"]],
            on=["seed", "model", "support_ratio"],
        )
    )
    counts = (
        df.groupby(["seed", "model", "support_ratio", "class_label"]).size()
        .rename("class_count")
        .reset_index()
        .pivot(
            index=["seed", "model", "support_ratio"],
            columns="class_label",
            values="class_count",
        )
        .rename(columns={0: "class0_count", 1: "class1_count"})
        .reset_index()
        .fillna(0)
    )
    merged = table.merge(counts, on=["seed", "model", "support_ratio"], how="left")
    merged.sort_values(["seed", "model", "support_ratio"], inplace=True)
    out_path = RESULTS_DIR / "classic_motif_summary.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved seed-level summary to {out_path}")


def compute_top_k_per_class(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    sorted_df = df.sort_values(
        ["seed", "support_ratio", "model", "class_label", "importance"],
        ascending=[True, True, True, True, False],
    )
    return (
        sorted_df.groupby(["seed", "support_ratio", "model", "class_label"])
        .head(top_k)
        .reset_index(drop=True)
    )


def save_top_motif_summary(top_df: pd.DataFrame, top_n: int = 3) -> None:
    agg = (
        top_df.groupby(["model", "feature_name", "class_label"])
        .agg(
            mean_importance=("importance", "mean"),
            appearances=("feature_name", "count"),
            seed_coverage=("seed", "nunique"),
            support_coverage=("support_ratio", "nunique"),
        )
        .reset_index()
    )
    rows = []
    for (model, class_label), group in agg.groupby(["model", "class_label"]):
        subset = group.sort_values(
            ["appearances", "mean_importance"], ascending=[False, False]
        ).head(top_n)
        rows.append(subset.assign(model=model, class_label=class_label))
    if rows:
        summary_df = pd.concat(rows, ignore_index=True)
        out_path = RESULTS_DIR / "classic_top_motif_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"Saved recurring-motif summary to {out_path}")


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "classic_motif_importances.csv")
    save_seed_level_summary(df)
    top_k_df = compute_top_k_per_class(df, TOP_K_PER_CLASS)
    top_k_path = RESULTS_DIR / "classic_top5_per_class.csv"
    top_k_df.to_csv(top_k_path, index=False)
    print(f"Saved top-{TOP_K_PER_CLASS} motifs per class to {top_k_path}")
    save_top_motif_summary(top_k_df, top_n=3)


if __name__ == "__main__":
    main()
