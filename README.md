# Project 3 – Large-Scale Supervised Graph Learning

Group 2 · Emily Massie, Aryan, Josh, Jashu, Kyle · Due 11/20/2025

---

## Introduction & Requirements

This repository contains our four deliverables for CS6010 Project 3 on the MUTAG dataset. The course brief asks us to:

1. Mine frequent subgraphs with gSpan, build classic ML pipelines with those motifs, and run ablations over supports/model parameters.
2. Train at least two GNN architectures, ablate their hyperparameters, and study efficiency/quality trade-offs.
3. Compare the classical and neural approaches.
4. Run explainability analyses for the best GNN models.

The MUTAG graphs (nitroaromatic compounds) live under `data/MUTAG/` in both raw `.txt` and processed `.pt` format; PyTorch Geometric downloads them automatically. Required packages (mirroring `pyrequirements.txt`) are:

- `torch>=2.9.1`
- `torch_geometric`
- `scikit-learn>=1.7.2`
- `ipykernel`
- `gspan-mining`
- `pandas`

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r pyrequirements.txt
```

To refresh the dataset cache at any time:

```bash
python data_download/download_mutag.py
```

This invokes `data_access/mutag.py`, which wraps `torch_geometric.datasets.TUDataset` and guarantees deterministic train/val/test splits via a seed parameter.

Deactivate the virtual environment with `deactivate` when finished.

---

## Repository Layout

- `data_download/download_mutag.py` – one-off script to populate `./data/MUTAG/`.
- `data_access/mutag.py` – shared loader returning seed-controlled dataset slices and PyG data loaders.
- `q1_frequent_subgraphs_classic_ml/` – gSpan mining, feature construction, classical model training, and packaging utilities.
- `q2_gnn/` – GCN + GIN models, training loops, and experiment runner.
- `q3_comparison/` – aggregation utilities + comparison plots/CSVs for Q3.
- `q4_explainability/` – GNNExplainer workflow and outputs for Q4.
- `data/` – cached MUTAG tensors (`raw/` + `processed/`).

---

## Methodology Overview

1. Download MUTAG via `TUDataset` once.
2. Decide on classical motifs vs GNN architectures and corresponding grids.
3. For each seed, run Q1 scripts in order (mine → build features → train classic models); each step only depends on the prior outputs.
4. For each seed, run Q2 experiments which load the same split and sweep hyperparameters per architecture.
5. Aggregate per-seed CSVs for ablation studies (to be automated in Q3).
6. Run explainability tooling (forthcoming) on the best GNN checkpoints.

---

## Dataset

Located in `./data/MUTAG/`:

- `raw/` – the original `.txt` files (graph indicator, edges, labels). We keep them for reproducibility and possible custom preprocessing.
- `processed/` – PyG tensors generated automatically; the scripts read from here via `torch_geometric`.

`data_access/mutag.load_mutag(...)` always shuffles with a provided seed before slicing the dataset into 80/10/10 train/val/test subsets, so every pipeline can line up their splits.

---

## Q1 – Frequent Subgraph Mining + Classical ML

Run the following (per-seed artifacts/features stay gitignored; package zipped outputs when needed):

```bash
python q1_frequent_subgraphs_classic_ml/frequent_subgraph_mining.py
python q1_frequent_subgraphs_classic_ml/construct_features.py --base-top-k-per-class 50
python q1_frequent_subgraphs_classic_ml/train_classic_models.py \
    --rf-config optional/path/to/rf_grid.json \
    --svm-config optional/path/to/svm_grid.json
python q1_frequent_subgraphs_classic_ml/package_outputs.py
```

Key details:

- `frequent_subgraph_mining.py` loops over the default seed list (currently `[0, 1, 2]`). For each seed, it mines motifs per class and per support ratio, writing JSONs under `artifacts/seed_<S>/class_<label>/support_XX/`.
- `construct_features.py` reads those artifacts, keeps a configurable top-K motifs per class (scaled by support), computes subgraph-isomorphism counts (or binary indicators), and stores the resulting matrices in `features/seed_<S>/support_XX/`. Each `feature_config.json` logs the seed, motif counts, runtime, and mode.
- `train_classic_models.py` loads each seed/support feature set, trains Random Forest and (linear/RBF) SVMs, and writes metrics to `results/seed_<S>/classic_ml_support_XX.csv`. Metrics include accuracy, macro precision/recall/F1, ROC-AUC, feature dimensionality, training time, and inference time.
- `package_outputs.py` zips each seed/support’s artifacts and features (e.g., `archives/artifacts_seed_0_support_0.10.zip`) so representative files can be shared without keeping the heavy directories in git.

Current snapshot (top-50 motifs per class, count features, single seed example):

| Support | Best Model | Val Acc | Test Acc | Val F1 | Test F1 | Train Time (s) |
|---------|------------|---------|----------|--------|---------|----------------|
| 0.10    | Linear SVM (C=0.1) | 1.00 | 0.80 | 1.00 | 0.762 | 0.16 |
| 0.20    | Linear SVM (C=0.1) | 1.00 | 0.80 | 1.00 | 0.762 | 0.14 |
| 0.30    | Linear SVM (C=0.1) | 0.94 | 0.80 | 0.935 | 0.780 | 0.14 |
| 0.40    | Random Forest (100 trees) | 0.89 | 0.75 | 0.862 | 0.715 | 0.10 |

Per-seed CSVs under `q1_frequent_subgraphs_classic_ml/results/` provide the full grids.

---

## Q2 – Graph Neural Networks (GCN & GIN)

Entry point:

```bash
python q2_gnn/run_experiments.py
```

Configuration constants inside the script define:

- `EXPERIMENT_SEEDS = [0, 1, 2]` (matching Q1 splits).
- Hyperparameter grid (`HIDDEN_DIMS`, `NUM_LAYERS`, `DROPOUTS`, `LRS`).
- `RUN_GCN`, `RUN_GIN`, epoch budget, and batch size.

For each seed and hyperparameter combo:

1. The model (GCN or GIN from `q2_gnn/models.py`) is instantiated with the chosen depth/width/dropout.
2. Trained for `NUM_EPOCHS` epochs via `train_epoch` (shared in `training.py`).
3. Evaluated once on validation/test loaders. `evaluate` returns accuracy, macro precision/recall/F1, ROC-AUC, plus inference time.
4. A result row is recorded with the seed, parameters, train/runtime stats, and metrics.

Outputs (per seed):

- `q2_gnn/results/seed_<S>/gcn_results.csv`
- `q2_gnn/results/seed_<S>/gin_results.csv`

The per-seed subdirectories mirror how Q1 stores features/results, making downstream aggregation straightforward. Explainability metrics (e.g., GNNExplainer) will join `q2_gnn/results/seed_<S>/` once the best configs are finalized.

---

## Q3 – Comparison

Run the aggregator to generate comparison tables/plots:

```bash
python q3_comparison/compare_q1_q2.py
```

The script:

1. Reads every per-seed CSV from Q1 (`q1_frequent_subgraphs_classic_ml/results/`) and Q2 (`q2_gnn/results/seed_*/`).
2. Merges them into `q3_comparison/aggregated_results.csv`, adding Q1 preprocessing time (mining + feature construction) so total pipeline time is comparable to the GNN training times.
3. Produces summary CSVs:
   - `summary_by_config.csv` – mean ± std across seeds for every model/support/hyperparameter combo.
   - `best_configs.csv` – best validation-accuracy config per model (RandomForest, LinearSVM, RBFSVM, GCN, GIN) with corresponding test metrics/efficiency numbers.
   - `summary_by_model.csv` – averages per individual model/config family.
   - `summary_by_family.csv` – high-level averages per family (Classic vs GCN vs GIN).
4. Saves per-metric bar charts grouped by model (classical supports are shown via separate line plots in `quality/classic_support_<metric>.png`) under `q3_comparison/figures/quality/` (e.g., `test_accuracy.png`, `test_f1.png`, `test_auc.png`) and per-metric efficiency plots under `q3_comparison/figures/efficiency/` (e.g., `train_time_sec.png`, `total_pipeline_time_sec.png`, `test_inference_time_sec.png`). Each plot aggregates every seed/support/config so classical vs. GNN models can be compared metric by metric without clutter.

Use these artifacts in the report or for further analysis notebooks under `q3_comparison/`.

---

## Q4 – Explainability

Run the explainer script to train the best GCN/GIN configurations, generate GNNExplainer masks, and log fidelity/sparsity/runtime metrics:

```bash
python q4_explainability/run_gnn_explainer.py \
    --edge-keep-ratios 0.2 0.3 0.5    # optional: sweep multiple sparsity targets
    --seeds 0 1 2                     # run explainability for each split seed
    --max-graphs -1                   # explain every correctly classified test graph
```

The script retrains the best configs from Q2 (GCN & GIN), identifies correctly classified test graphs, and uses PyG’s `GNNExplainer` to compute fidelity⁺, fidelity⁻, sparsity, and runtime for each explanation. Metrics are saved per model under:

- `q4_explainability/results/GCN/gnn_explainer_metrics.csv`
- `q4_explainability/results/GIN/gnn_explainer_metrics.csv`

and also aggregated into `q4_explainability/results/gnn_explainer_metrics.csv` for quick comparison.

To document the “self-explainable” classical side we added:

```bash
python q4_explainability/classic_motif_explainability.py  # optional: customize --seeds, --supports, --top-k
```

This utility reuses the Q1 feature tensors and the best hyperparameters recorded in `q1_frequent_subgraphs_classic_ml/results/` to retrain Random Forest and Linear SVM models, then exports the top‑K motifs per seed/support/model (feature ranks, class labels, support counts, signed coefficients, and the actual subgraph structure) to `q4_explainability/results/classic_motif_importances.csv`. Those importance lists satisfy the “self-explainable” requirement for classical ML and can be read alongside the GNNExplainer metrics when discussing fidelity vs. sparsity.

With both artifacts in hand we directly compare post-hoc GNN explanations (fidelity⁺/⁻, sparsity, runtime) against the intrinsic motif-level explanations from the classical pipeline. Run `python q4_explainability/plot_explainability.py` to regenerate the figures under `q4_explainability/figures/`, which now include per-seed breakdowns:

- `figures/gnn/fidelity_sufficiency_necessity.png`: shows, for each GCN/GIN seed, how much confidence remains when we keep only the explainer mask (fidelity⁺) versus how much confidence drops when we remove it (fidelity⁻). This directly captures the “sufficiency vs. necessity” discussion required in Q4.
- `figures/gnn/*_per_seed.png`: line/bar charts for fidelity⁺, fidelity⁻, sparsity, and runtime per seed so we can see how each split behaves.
- `figures/classic/importance_vs_support.png` and `per_seed_importance_vs_support.png`: illustrate how the top‑K motif importance changes as the mining support varies (overall and per seed) for RandomForest and Linear SVM.
- `figures/classic/class_label_distribution*.png`: counts of class‑0 vs class‑1 motifs among the top-K importance lists, highlighting which class’s substructures dominate the explanations.
- `figures/classic/classic_vs_gnn_comparison.png`: ties the two worlds together by plotting mean classical motif importance against mean GNN fidelity⁺ per seed.

---

## Shared Results & Notebooks

- The repository root `results/` directory collects lightweight aggregation tables exported for the report (`classic_ml_all_runs.csv`, `classic_ml_ablation_results.csv`, `gnn_main_results.csv`, `classic_vs_gnn_summary.csv`, etc.). Use these when you only need the distilled metrics without re-running Q1/Q2/Q3.
- The notebook `Project 3 Classical and GNN v2.ipynb` mirrors the scripted pipelines in an interactive format. It contains exploratory visualizations, trains additional GraphSAGE baselines, and can be used to regenerate illustrative plots (but the scripts above remain the canonical workflow).

---

## Next Steps

1. Re-run Q1 and Q2 pipelines to populate the new per-seed directories and CSVs.
2. Add Q3 aggregation scripts to summarize the classical vs. GNN ablations.
3. Integrate GNN explainability workflows into Q4.
4. Consider exporting trained checkpoints or additional metadata if Q3/Q4 need them.

With the shared seed-aware loader in place, each question can evolve independently while guaranteeing consistent, reproducible splits across experiments.


# Results and Conclusions

The programs for both the dataset runs and build out ML models. The outputs for each program were used for our results, which can all be seen in our report.
