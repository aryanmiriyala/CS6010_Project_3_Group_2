import copy
import itertools
import statistics
import time
from typing import Dict, List, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.explain import Explanation, Explainer, GNNExplainer, ModelConfig
from torch_geometric.explain.metric import fidelity
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from data_access.mutag import DataLoaders, DatasetSplits, load_mutag


BATCH_SIZE = 32
SPLITS = (0.8, 0.1, 0.1)
EPOCHS = 50
HIDDEN_DIMS = [32, 64]
NUM_LAYERS = [2, 3, 4]
DROPOUTS = [0.0, 0.2, 0.75]
LRS = [0.001, 0.005]


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.lin(x)


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()

        mlp = lambda in_dim: torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(mlp(num_features)))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(mlp(hidden_dim)))

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, preds = [], []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        ys.append(data.y)
        preds.append(pred)

    y = torch.cat(ys).cpu()
    p = torch.cat(preds).cpu()

    return accuracy_score(y, p), f1_score(y, p, average="macro")


def run_single_model(
    model_class: Type[torch.nn.Module],
    name: str,
    loaders: DataLoaders,
    num_features: int,
    num_classes: int,
) -> List[Dict]:
    results: List[Dict] = []

    for h, L, d, lr in itertools.product(HIDDEN_DIMS, NUM_LAYERS, DROPOUTS, LRS):
        model = model_class(num_features, num_classes, hidden_dim=h, num_layers=L, dropout=d)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val = 0.0
        best_test = 0.0

        start_time = time.time()
        for _ in range(EPOCHS):
            train_epoch(model, optimizer, loaders.train)
            val_acc, _ = evaluate(model, loaders.val)
            if val_acc > best_val:
                best_val = val_acc
                best_test = evaluate(model, loaders.test)[0]

        runtime = time.time() - start_time
        results.append(
            {
                "Model": name,
                "HiddenDim": h,
                "Layers": L,
                "Dropout": d,
                "LR": lr,
                "ValAcc": best_val,
                "TestAcc": best_test,
                "RuntimeSec": runtime,
            }
        )

        print(
            f"{name} | h={h} L={L} d={d} lr={lr} → "
            f"Val={best_val:.3f} Test={best_test:.3f} Time={runtime:.1f}s"
        )

    return results


def get_trained_model(
    model_class: Type[torch.nn.Module],
    config: Dict,
    loaders: DataLoaders,
    num_features: int,
    num_classes: int,
):
    model = model_class(
        num_features,
        num_classes,
        hidden_dim=config["HiddenDim"],
        num_layers=config["Layers"],
        dropout=config["Dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    best_val = 0.0
    best_state = None

    for _ in range(EPOCHS):
        train_epoch(model, optimizer, loaders.train)
        val_acc, _ = evaluate(model, loaders.val)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def explain_model(model, loader, model_name, max_graphs=5):
    model.eval()
    model_config = ModelConfig(
        mode="multiclass_classification", task_level="graph", return_type="raw"
    )
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config,
    )

    fid_plus_list: List[float] = []
    fid_minus_list: List[float] = []
    sparsity_list: List[float] = []
    runtime_list: List[float] = []
    explained_count = 0

    for data in loader:
        if explained_count >= max_graphs:
            break

        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1).item()
        if pred != data.y.item():
            continue

        start_time = time.time()
        explanation: Explanation = explainer(data.x, data.edge_index, batch=data.batch)
        runtime = time.time() - start_time
        fid_minus, fid_plus = fidelity(explainer, explanation)

        edge_sparsity = 1 - explanation.edge_mask.sigmoid().mean().item()
        node_mask = explanation.get("node_mask", None)
        node_sparsity = (
            1 - node_mask.sigmoid().mean().item() if node_mask is not None else 0.0
        )
        sparsity = (edge_sparsity + node_sparsity) / 2

        fid_plus_list.append(fid_plus)
        fid_minus_list.append(fid_minus)
        sparsity_list.append(sparsity)
        runtime_list.append(runtime)
        explained_count += 1

    if explained_count == 0:
        print(f"\nNo correct predictions available for explanation for {model_name}.")
        return None

    print(f"\n{model_name} Explainability Metrics (avg over {explained_count} graphs):")
    print(f"Fidelity+: {statistics.mean(fid_plus_list):.4f}")
    print(f"Fidelity-: {statistics.mean(fid_minus_list):.4f}")
    print(f"Sparsity: {statistics.mean(sparsity_list):.4f}")
    print(f"Runtime (sec): {statistics.mean(runtime_list):.4f}")

    return {
        "Fidelity+": statistics.mean(fid_plus_list),
        "Fidelity-": statistics.mean(fid_minus_list),
        "Sparsity": statistics.mean(sparsity_list),
        "Runtime": statistics.mean(runtime_list),
    }


def run_experiments():
    dataset, splits, loaders = load_mutag(batch_size=BATCH_SIZE, splits=SPLITS)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    print("\n================ GCN Experiments ================\n")
    gcn_results = run_single_model(GCN, "GCN", loaders, num_features, num_classes)

    print("\n================ GIN Experiments ================\n")
    gin_results = run_single_model(GIN, "GIN", loaders, num_features, num_classes)

    all_results = gcn_results + gin_results
    sorted_results = sorted(all_results, key=lambda x: x["TestAcc"], reverse=True)
    for r in sorted_results[:10]:
        print(r)

    return dataset, splits, loaders, gcn_results, gin_results


def main():
    dataset, splits, loaders, gcn_results, gin_results = run_experiments()

    # Explainability step
    explain_loader = DataLoader(splits.test, batch_size=1, shuffle=False)

    if gcn_results:
        best_gcn = max(gcn_results, key=lambda x: x["TestAcc"])
        best_gcn_model = get_trained_model(GCN, best_gcn, loaders, dataset.num_features, dataset.num_classes)
        explain_model(best_gcn_model, explain_loader, "Best GCN")

    if gin_results:
        best_gin = max(gin_results, key=lambda x: x["TestAcc"])
        best_gin_model = get_trained_model(GIN, best_gin, loaders, dataset.num_features, dataset.num_classes)
        explain_model(best_gin_model, explain_loader, "Best GIN")


if __name__ == "__main__":
    main()
