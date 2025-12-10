#!/usr/bin/env python3
"""Summarize classical explainability metrics into a table."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "classic_motif_importances.csv")
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
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
