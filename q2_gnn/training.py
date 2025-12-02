import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
    start = time.time()
    ys, preds, outs = [], [], []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        ys.append(data.y)
        preds.append(pred)
        outs.append(out)

    inference_time = time.time() - start

    y = torch.cat(ys).cpu()
    p = torch.cat(preds).cpu()
    o = torch.cat(outs).cpu()

    # Map MUTAG labels {-1, 1} to {0, 1} for sklearn metrics.
    y_mapped = y.clone()
    y_mapped[y_mapped == -1] = 0

    probs = F.softmax(o, dim=1)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_mapped, p),
        "precision": precision_score(y_mapped, p, average="macro", zero_division=0),
        "recall": recall_score(y_mapped, p, average="macro", zero_division=0),
        "f1": f1_score(y_mapped, p, average="macro", zero_division=0),
        "auc": roc_auc_score(y_mapped, probs),
    }

    return metrics, inference_time
