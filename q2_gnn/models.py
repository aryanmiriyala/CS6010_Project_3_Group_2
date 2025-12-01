import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim=64, num_layers=3, dropout=0.2):
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
    def __init__(self, num_features: int, num_classes: int, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()

        # This MLP is for the input layer
        self.in_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # This MLP is for the hidden layers
        hidden_mlp = lambda: torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(self.in_mlp))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(hidden_mlp()))

        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.lin(x)
