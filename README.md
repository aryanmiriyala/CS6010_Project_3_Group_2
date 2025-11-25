# Project 3 – Large-Scale Supervised Graph Learning

Group 2 · Emily Massie, Aryan, Josh, Jashu, Kyle · Due 11/20/2025

This repo houses our four deliverables on the MUTAG dataset:

1. **Q1 – Frequent Subgraph Mining + Classic ML** (`q1_frequent_subgraphs_classic_ml/`)
2. **Q2 – Graph Neural Networks** (`q2_gnn/`)
3. **Q3 – Comparison of the two approaches** (`q3_comparison/`)
4. **Q4 – Explainability for the GNN models** (`q4_explainability/`)

Shared utilities and the MUTAG download script live at the repo root so each workstream can stay focused on its pipelines.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r pyrequirements.txt
```

### Download / Cache MUTAG

We now keep dataset handling in one place. Run this once (or whenever you need to refresh the cache):

```bash
python data_download/download_mutag.py
```

This script calls `data_access/mutag.py`, which uses `torch_geometric.datasets.TUDataset` to pull the dataset into `./data/MUTAG/`.

---

## Repository Layout

- `data_download/download_mutag.py` – convenience entrypoint to fetch MUTAG before running experiments.
- `data_access/mutag.py` – shared loader returning consistent train/val/test splits and PyG data loaders for every pipeline.
- `q1_frequent_subgraphs_classic_ml/` – hosts the standalone mining/classic ML pipeline (see `frequent_subgraph_mining.py`). Generated outputs (`artifacts/`, `features/`) stay gitignored because the mined pattern JSONs can exceed GitHub’s file-size limits; regenerate them locally and, when needed for sharing, compress them via `package_outputs.py` which writes `.zip` archives under `q1_frequent_subgraphs_classic_ml/archives/`.
- `q2_gnn/` – contains `gnn.py`, now importing the shared loader to build/train GCN & GIN models, run the hyperparameter ablations, and kick off explainability passes for the best runs.
- `q3_comparison/`, `q4_explainability/` – reserved for scripts/notebooks that will aggregate metrics and run post-hoc explainers once Q1 and Q2 output standardized logs.
- `data/` – houses the MUTAG raw/processed tensors downloaded by PyG (kept for reproducibility).

---

## Running the Existing Pipelines

- **Classical (Q1):**
  ```bash
  python q1_frequent_subgraphs_classic_ml/frequent_subgraph_mining.py
  python q1_frequent_subgraphs_classic_ml/construct_features.py --base-top-k-per-class 50
  python q1_frequent_subgraphs_classic_ml/train_classic_models.py \
      --rf-config optional/path/to/rf_grid.json \
      --svm-config optional/path/to/svm_grid.json
  python q1_frequent_subgraphs_classic_ml/package_outputs.py
  ```
  The first command mines frequent subgraphs per class (support thresholds configurable via `--support-thresholds`) and drops artifacts into `q1_frequent_subgraphs_classic_ml/artifacts/` (ignored in git). The second automatically scans those artifacts, scales the number of retained motifs per support ratio (`--base-top-k-per-class` controls the cap; pass `--disable-ratio-scaling` or `0` to keep all) and writes motif-count feature matrices for train/val/test under `q1_frequent_subgraphs_classic_ml/features/` (add `--binary-features` if you only want presence/absence). Each `feature_config.json` now logs how many motifs were kept plus the time it took to build that feature set. The third trains Random Forest & SVM baselines across every feature set and saves accuracy/precision/recall/F1 along with training + inference runtimes and feature dimensionality to `q1_frequent_subgraphs_classic_ml/results/`. Finally, `package_outputs.py` compresses each support ratio’s artifacts/features into `.zip` files in `q1_frequent_subgraphs_classic_ml/archives/` so we can check in representative outputs without relying on Git LFS.

  **Current Q1 metrics (top-50 motifs per class, count features):**

  | Support | Best Model | Val Acc | Test Acc | Val F1 | Test F1 | Train Time (s) |
  |---------|------------|---------|----------|--------|---------|----------------|
  | 0.10    | Linear SVM (C=0.1) | 1.00 | 0.80 | 1.00 | 0.762 | 0.16 |
  | 0.20    | Linear SVM (C=0.1) | 1.00 | 0.80 | 1.00 | 0.762 | 0.14 |
  | 0.30    | Linear SVM (C=0.1) | 0.94 | 0.80 | 0.935 | 0.780 | 0.14 |
  | 0.40    | Random Forest (100 trees) | 0.89 | 0.75 | 0.862 | 0.715 | 0.10 |

  Raw metrics (including precision/recall, inference time, and exact hyperparameters) live in `q1_frequent_subgraphs_classic_ml/results/classic_ml_support_XX.json` for each support ratio.
- **GNNs (Q2):**
  ```bash
  python q2_gnn/gnn.py
  ```
  The script now:
  - pulls MUTAG splits via `data_access.mutag.load_mutag(...)`
  - trains GCN & GIN over the configured hyperparameter grid
  - prints per-config validation/test accuracy and runtime
  - retrains the best configs and runs GNNExplainer on a few correctly classified graphs

As other questions are implemented, keep their code (trainers, evaluation scripts, explainability notebooks) inside the respective `Q*_.../` directories so each deliverable stays isolated yet consistent via the shared data utilities.

---

## Next Steps

- Extend `q1_frequent_subgraphs_classic_ml` beyond the initial gSpan run: persist mined motifs, build feature matrices, and add classical models plus ablation sweeps (scripts now exist; rerun them locally as outputs remain gitignored).
- Teach `q2_gnn/gnn.py` to export metrics/checkpoints to feed into Q3 comparisons.
- Populate `q3_comparison/` and `q4_explainability/` with scripts that ingest the standardized outputs and produce tables/plots for the report.

With the shared dataset loader and download script in place, every new piece of code can assume MUTAG resides in `./data` and that the splits/loaders stay consistent across experiments.
