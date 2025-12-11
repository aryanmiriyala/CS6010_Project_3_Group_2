# Project 3 – Group 2

This is the README file for Project 3 of CS 6010 Data Science Programming.
Written by: Emily Massie, Aryan Miriyala, Joshua Gabriel, Jashhvanth Tamilselvan Kunthavai, and Kyle Cusimano  
Due 12/11/2025

# Introduction and Requirements

This repository holds the base code for our classical pipelines, GNN pipelines, dataset download helper, and generated outputs for the MUTAG mutagenicity task. The goal is to mine frequent subgraphs, train classical models on those motifs, train GCN/GIN baselines on the same splits, compare both families, and analyze explanations. All experiments use the MUTAG dataset (188 nitroaromatic molecular graphs) from PyTorch Geometric or Hugging Face (https://huggingface.co/datasets/graphs-datasets/MUTAG). The datasets are downloaded into `data/MUTAG/` via the provided scripts and are gitignored.

The processing code requires:

```
python 3.11+
torch>=2.9.1
torch_geometric>=2.7.0
scikit-learn>=1.7.2
gspan-mining
pandas
matplotlib
seaborn
numpy
networkx
ipykernel
```

# File Path Locations

## data_download

- `download_mutag.py` fetches MUTAG (raw and processed) and places it under `data/MUTAG/`.

## data_access

- `mutag.py` exposes helper functions to build deterministic train/val/test splits and PyG dataloaders shared by every script.

## q1_frequent_subgraphs_classic_ml

- `frequent_subgraph_mining.py` mines motifs per class/support.
- `construct_features.py` converts motifs to feature matrices.
- `train_classic_models.py` trains Random Forest and SVM baselines.
- `package_outputs.py` zips artifacts/features for sharing.
- Outputs (artifacts, features, results, archives) are created locally after running the scripts.

## q2_gnn

- `run_experiments.py` trains both GCN and GIN models across depth/width/dropout/lr grids.
- `models.py` defines architectures.
- `training.py` and `result_graphs.py` include the training loops and plotting helpers.
- CSV results per seed/model appear under `q2_gnn/results/` after execution.

## q3_comparison

- `compare_q1_q2.py` aggregates every classical and GNN run, exporting summary tables and figures under `q3_comparison/`.

## q4_explainability

- `run_gnn_explainer.py` retrains the best GCN/GIN configs and collects GNNExplainer metrics.
- `classic_motif_explainability.py` reports the top classical motifs.
- `plot_explainability.py` regenerates fidelity/sparsity/importances figures.

## data, q\*/results/, results/

- Generated datasets, CSV metrics, plots, and archives live inside these directories once the scripts are executed. They are gitignored but referenced throughout the report.

## Project 3 Classical and GNN v2.ipynb

- Notebook version of the scripted workflows for exploratory analysis.

# Methodology

1. Create and activate a virtual environment.
2. Install the dependencies from `pyrequirements.txt`.
3. Run `python data_download/download_mutag.py` to populate `data/MUTAG/`.
4. Execute the Q1 scripts in order (mining → features → training → packaging) for classical baselines.
5. Execute `python q2_gnn/run_experiments.py` to collect neural baselines.
6. Run `python q3_comparison/compare_q1_q2.py` to merge all results and export summary visuals.
7. Run the Q4 scripts to produce explainability metrics for both classical and GNN models.

# Setup and Execution

## Virtual Environment

macOS/Linux

```
python -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip
pip install -r pyrequirements.txt
```

Windows PowerShell

```
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r pyrequirements.txt
```

## Running the Pipelines

All commands are launched from the project root with the environment active.

```
python data_download/download_mutag.py
python q1_frequent_subgraphs_classic_ml/frequent_subgraph_mining.py
python q1_frequent_subgraphs_classic_ml/construct_features.py
python q1_frequent_subgraphs_classic_ml/train_classic_models.py
python q1_frequent_subgraphs_classic_ml/package_outputs.py
python q2_gnn/run_experiments.py
python q3_comparison/compare_q1_q2.py
python q4_explainability/run_gnn_explainer.py
python q4_explainability/classic_motif_explainability.py
python q4_explainability/plot_explainability.py
```

Outputs include CSV metrics, PNG plots, archived artifacts, and explainability summaries under the respective module folders. Consult those generated directories along with `results/` and the notebook for the analyses used in the final report.
