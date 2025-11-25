"""
Utility functions for downloading and loading the MUTAG dataset in a single place.

This module centralizes the logic so every question-specific pipeline can work from
the same data splits and loader configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


@dataclass(frozen=True)
class DatasetSplits:
    """Container for the train/val/test slices returned by ``TUDataset``."""

    train: TUDataset
    val: TUDataset
    test: TUDataset


@dataclass(frozen=True)
class DataLoaders:
    """Container for the PyG data loaders that iterate over the split datasets."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def ensure_mutag_downloaded(data_root: str | Path = DEFAULT_DATA_ROOT) -> int:
    """
    Trigger the download/processing of MUTAG (if not already cached) and return
    the number of graphs available.
    """

    dataset = TUDataset(root=str(data_root), name="MUTAG")
    return len(dataset)


def load_mutag(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    splits: Sequence[float] = (0.8, 0.1, 0.1),
    shuffle: bool = True,
) -> Tuple[TUDataset, DatasetSplits, DataLoaders]:
    """
    Download (if needed) and load the MUTAG dataset, returning both dataset slices
    and data loaders so that each pipeline can share identical inputs.
    """

    if len(splits) != 3:
        raise ValueError("splits must contain exactly three ratios (train, val, test).")

    total_ratio = sum(splits)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("splits must sum to 1.0; received %.4f" % total_ratio)

    dataset = TUDataset(root=str(data_root), name="MUTAG")
    if shuffle:
        dataset = dataset.shuffle()

    total = len(dataset)
    train_len = int(splits[0] * total)
    val_len = int(splits[1] * total)
    test_len = total - train_len - val_len

    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len : train_len + val_len]
    test_dataset = dataset[train_len + val_len :]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return dataset, DatasetSplits(train_dataset, val_dataset, test_dataset), DataLoaders(
        train_loader, val_loader, test_loader
    )


if __name__ == "__main__":
    num_graphs = ensure_mutag_downloaded()
    print(f"MUTAG is ready with {num_graphs} graphs cached under {DEFAULT_DATA_ROOT}/MUTAG")
