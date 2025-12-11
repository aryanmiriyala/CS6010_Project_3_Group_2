# Project 3 – Large-Scale Supervised Graph Learning

Group 2 · CS 6010 Fall 2025 · MUTAG binary classification

## Stack
- Python 3.11+
- `pip install -r pyrequirements.txt` (PyTorch, PyTorch Geometric, scikit-learn, gspan-mining, pandas/matplotlib/seaborn, numpy, networkx)

## Setup
```bash
python -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r pyrequirements.txt
python data_download/download_mutag.py  # (re)builds ./data/MUTAG/
```

## Data Access
`data_access/mutag.py` wraps `torch_geometric.datasets.TUDataset`, applies a deterministic 80/10/10 split per seed, and returns PyG dataloaders for every script below. All pipelines expect the cached tensors in `data/MUTAG/{raw,processed}`.

### Dataset Snapshot
- **Source:** MUTAG (TU Dortmund graph benchmark) downloaded via PyG’s `TUDataset` or directly from Hugging Face: https://huggingface.co/datasets/graphs-datasets/MUTAG.
- **Contents:** 188 nitroaromatic compounds, binary labels (mutagenic/non-mutagenic), node attributes encoding atom type, edge attributes encoding bond type.
- **Location:** `data/MUTAG/raw/` stores the original `.txt` indicator/graph/label files; `data/MUTAG/processed/` holds PyG tensors regenerated on demand.
- **Splits:** Every script uses the same seeded 80/10/10 train/val/test division from `data_access/mutag.load_mutag`.

## Pipelines
Each question has a single entry script; run them with the virtual environment active. All scripts default to seeds `[0,1,2]` and reuse the same splits.

### Q1 – Frequent Subgraphs + Classical ML
```bash
python q1_frequent_subgraphs_classic_ml/frequent_subgraph_mining.py
```
Discovers class-specific motifs with gSpan and writes them under `artifacts/seed_*`.

```bash
python q1_frequent_subgraphs_classic_ml/construct_features.py
```
Converts mined motifs into feature matrices (counts/indicators) stored in `features/seed_*`.

```bash
python q1_frequent_subgraphs_classic_ml/train_classic_models.py
```
Trains Random Forest plus linear/RBF SVMs on every seed/support pair and logs metrics in `results/seed_*/classic_ml_support_XX.csv`.

```bash
python q1_frequent_subgraphs_classic_ml/package_outputs.py
```
Default flow mines motifs, builds feature matrices with the top motifs per class, trains Random Forest + SVM baselines, and zips artifacts for sharing. Outputs live in `artifacts/`, `features/`, `results/seed_*/`, and `archives/`.

### Q2 – Graph Neural Networks (GCN & GIN)
```bash
python q2_gnn/run_experiments.py
```
Runs both GCN and GIN sweeps (hidden dims, layers, dropout, learning rate) and writes per-seed CSVs to `q2_gnn/results/seed_*/{gcn,gin}_results.csv`; `result_graphs.py` rebuilds the ablation figures.

### Q3 – Classical vs. GNN Comparison
```bash
python q3_comparison/compare_q1_q2.py
```
Merges all Q1/Q2 CSVs, computes summary/aggregate tables, and exports both the combined metrics and bar charts (`aggregated_results.csv`, `summary_by_{config,model,family}.csv`, `best_configs.csv`, `figures/`).

### Q4 – Explainability
```bash
python q4_explainability/run_gnn_explainer.py
```
Retrains the best GCN/GIN configs, generates GNNExplainer masks, and logs fidelity/sparsity/runtime metrics per test graph.

```bash
python q4_explainability/classic_motif_explainability.py
```
Re-trains the best classical models, ranks the most important motifs, and saves CSVs per seed/model.

```bash
python q4_explainability/plot_explainability.py
```
By default the explainer retrains the best GCN/GIN configs, generates fidelity/sparsity metrics for every correctly classified test graph, extracts classical motif importances, and refreshes the plots under `q4_explainability/results/` and `q4_explainability/figures/`.

## Results Artifacts
- `results/` (repo root) – filtered CSVs used in the report (classic vs. GNN summaries, ablation tables, etc.).
- `Project 3 Classical and GNN v2.ipynb` – optional notebook that mirrors the scripted runs and contains spot-check visualizations.

Once the scripts finish, consult the `results/` and `q*/results/` directories or the associated figures for the data referenced in the final write-up.
