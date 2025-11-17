import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score
import itertools


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset = TUDataset(root="data", name="MUTAG")
dataset = dataset.shuffle()

# 80/10/10 split
train_len = int(0.8 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len

train_dataset = dataset[:train_len]
val_dataset   = dataset[train_len:train_len + val_len]
test_dataset  = dataset[train_len + val_len:]

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

num_features = dataset.num_features
num_classes  = dataset.num_classes


# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # input layer
        self.convs.append(GCNConv(num_features, hidden_dim))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)  # graph-level embedding
        return self.lin(x)


class GIN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()

        mlp = lambda: torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ))
        )

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(mlp()))

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.lin(x)


# ============================================================
# 3. TRAINING UTILITIES
# ============================================================

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


# ============================================================
# 4. EXPERIMENT LOOP (Ablations)
# ============================================================

hidden_dims  = [32, 64]
num_layers   = [2, 3, 4]
dropouts     = [0.0, 0.2]
lrs          = [0.001, 0.005]

results = []

def run_experiment(ModelClass, name):
    for h, L, d, lr in itertools.product(hidden_dims, num_layers, dropouts, lrs):

        model = ModelClass(hidden_dim=h, num_layers=L, dropout=d)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()
        best_val = 0
        best_test = 0

        # training loop
        for epoch in range(50):  
            train_epoch(model, optimizer, train_loader)
            val_acc, _ = evaluate(model, val_loader)
            if val_acc > best_val:
                best_val = val_acc
                best_test = evaluate(model, test_loader)[0]

        runtime = time.time() - start_time

        results.append({
            "Model": name,
            "HiddenDim": h,
            "Layers": L,
            "Dropout": d,
            "LR": lr,
            "ValAcc": best_val,
            "TestAcc": best_test,
            "RuntimeSec": runtime
        })

        print(f"{name} | h={h} L={L} d={d} lr={lr} → "
              f"Val={best_val:.3f} Test={best_test:.3f} Time={runtime:.1f}s")


# ============================================================
# 5. RUN BOTH MODELS
# ============================================================

print("\n================ GCN Experiments ================\n")
run_experiment(GCN, "GCN")

print("\n================ GIN Experiments ================\n")
run_experiment(GIN, "GIN")


# ============================================================
# 6. Show final sorted results
# ============================================================

sorted_results = sorted(results, key=lambda x: x["TestAcc"], reverse=True)
for r in sorted_results[:10]:
    print(r)
