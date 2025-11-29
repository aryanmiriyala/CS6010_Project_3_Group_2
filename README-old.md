# Title: Project 3 - Large Scale Supervised Machine Learning

This is the README file for Project 3 of CS 6010 Data Science Programming

# Group 2 {Emily Massie, Aryan, Josh, Jashu, Kyle}

---

# Introduction and Requirements

Contained in this series of folders are the base code for the programs executed and the datasets as txts and pt files. The main goal for this project was to construct feature vectors using gSpan, select two classical ML and GNN algorithms, run the MUTAG dataset through each algorithm (using the feature vectors for the classical ML algorithms), run ablation studies on each algorithm, compare classical ML algorithms with GNN algorithms and understand their similarities and differences, and run an explaination method to evaluate explanation quality. The MUTAG dataset that we used is mainly used for machine learning tasks. They are specifically a collection of nitroaromatic compounds.

The code we provided requires: - torch>=2.9.1 - torch_geometric - scikit-learn>=1.7.2 - ipykernel - gspan-mining - pandas

# File Path Locations

## data\MUTAG

    - Contains the actual MUTAG dataset used for the code

    - It is split into two parts: processed and raw. We opted to use the raw format (which are all txt files) for this project

## GNN

    - This contains the code for the GNN portion of the project.

    - The program file for creating the GNN's, running ablation studies, and running the explanation method can be found in /CS6010_Project_3_Group_2/GNN/gnn.py

## Classical_ML

    - This contains the code for the Classical ML portion of the project.

    - The program file for creating the classical ML algorithms and running ablation studies can be found in /CS6010_Project_3_Group_2/Classical_ML/ml_classy.ipynb.

# MUTAG Properties

# Research Questions

# Methodology

The main tool we used for this project was Python. There were multiple libraries that were used.

Below are all the steps we took to getting our results:

    1. Download the datasets from their respective links using TUDataset
    2. Made sure that all the required libraries are installed
    3. We decided on the GNN skews and classical ML algorithms that we will use
    4. Once this was done, some research was done to make sure that we knew what we were doing when it came building these ML algorithms
    5. After a brief research period, we were able to execute the models, conduct ablation studies, and make explainers (for GNN)
    6. Once this was done, we used the output from the programs and looked into them for analysis and extrapolated results

# Dataset

Located in '/data' folder
Contains the raw and processed files for the MUTAG dataset.

# (Q1) Frequent Subgraph mining & Classic ML Models

# (Q2) Graph Neural Network (GNNs)

Located in 'gnn.py'

## GCN

## GIN

# (Q3) Comparison

# (Q4) Explainability

# Setup and Execution

## Virtual Enviroment

python -m venv venv

### Activate Virtual Enviroment

#### windows (Powershell)

    venv\Scripts\Activate.ps1

#### windows (CMD)

    venv\Scripts\activate.bat

#### Linux

    source venv/activate

### Install Requirments

    pip install -r pyrequirments

#### Deactivate Envirioment

    deactivate

# Results and Conclusions

The programs for both the datasets run. The execution time was measured for each program and can be found above. The outputs for each program (including the PNGs and the TXT reports) were used to analyze, compare, and contrast the two graph network datasets.

## Run (from root, with venv active)

First, you must cd into the respective folders to execute the Python programs. The steps to do so can be seen below: - To execute the gnn.py file, use the following command: cd GNN - To execute the ml_classy.ipynb file, use the following command: cd Classical ML

Once you cd into the respective folders, you can use one of the two bash commands below to run the respective files:

    ```bash
    python gnn.py
    python ml_classy.ipynb
    ```

# Results and Conclusions

The programs for both the dataset runs and build out ML models. The outputs for each program were used for our results, which can all be seen in our report.
