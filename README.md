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
- `q1_frequent_subgraphs_classic_ml/` – hosts the standalone mining/classic ML pipeline (see `frequent_subgraph_mining.py`).
- `q2_gnn/` – contains `gnn.py`, now importing the shared loader to build/train GCN & GIN models, run the hyperparameter ablations, and kick off explainability passes for the best runs.
- `q3_comparison/`, `q4_explainability/` – reserved for scripts/notebooks that will aggregate metrics and run post-hoc explainers once Q1 and Q2 output standardized logs.
- `data/` – houses the MUTAG raw/processed tensors downloaded by PyG (kept for reproducibility).

---

## Running the Existing Pipelines

- **Classical (Q1):**
  ```bash
  python q1_frequent_subgraphs_classic_ml/frequent_subgraph_mining.py
  python q1_frequent_subgraphs_classic_ml/construct_features.py
  ```
  The first command mines frequent subgraphs per class (support thresholds configurable via `--support-thresholds`) and drops artifacts into `q1_frequent_subgraphs_classic_ml/artifacts/`. The second command converts the mined patterns into binary feature matrices for train/val/test under `q1_frequent_subgraphs_classic_ml/features/`, ready for classic model training.
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

- Extend `q1_frequent_subgraphs_classic_ml` beyond the initial gSpan run: persist mined motifs, build feature matrices, and add classical models plus ablation sweeps.
- Teach `q2_gnn/gnn.py` to export metrics/checkpoints to feed into Q3 comparisons.
- Populate `q3_comparison/` and `q4_explainability/` with scripts that ingest the standardized outputs and produce tables/plots for the report.

With the shared dataset loader and download script in place, every new piece of code can assume MUTAG resides in `./data` and that the splits/loaders stay consistent across experiments.
