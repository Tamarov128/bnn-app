"""
core/data/datasets.py
───────────────────────
DataManager: centralised dataset and DataLoader factory.

Supported datasets
──────────────────
  "mnist"         →  torchvision.datasets.MNIST
  "fashion_mnist" →  torchvision.datasets.FashionMNIST
  "kmnist"        →  core.data.kmnist.KMNISTDataset   (HuggingFace-backed)
  "omniglot"      →  torchvision.datasets.Omniglot
  "not_mnist"     →  core.data.notmnist.NotMNISTDataset (HuggingFace-backed)

TRAIN_DATASETS
──────────────
Registry used by the GUI to build the training-dataset selector panel.
Each entry carries the display name, the dataset class, and the constructor
kwargs for each split.

is_downloaded(name) / download_dataset(name)
────────────────────────────────────────────
Helpers called by the GUI's dataset panel to check download status and
trigger on-demand downloads without running a full training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets as tv_datasets

from core.config import ExperimentConfig, DATASETS_DIR, AVAILABLE_DATASETS
from core.data.transforms import (
    get_mnist_transform,
    get_omniglot_transform,
    get_notmnist_transform,
)
from core.data.kmnist import KMNISTDataset
from core.data.notmnist import NotMNISTDataset

_SPLIT_SEED = 42

# ── Dataset descriptor ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetInfo:
    key:          str
    display_name: str
    cls:          type
    train_kwargs: dict[str, Any]
    test_kwargs:  dict[str, Any]
    # Path relative to DATASETS_DIR that confirms the dataset is on disk.
    check_path:   str | None


# ── Registry ───────────────────────────────────────────────────────────────────

TRAIN_DATASETS: dict[str, DatasetInfo] = {

    "mnist": DatasetInfo(
        key          = "mnist",
        display_name = "MNIST",
        cls          = tv_datasets.MNIST,
        train_kwargs = {"train": True},
        test_kwargs  = {"train": False},
        check_path   = "MNIST/raw/train-images-idx3-ubyte",
    ),

    "fashion_mnist": DatasetInfo(
        key          = "fashion_mnist",
        display_name = "Fashion-MNIST",
        cls          = tv_datasets.FashionMNIST,
        train_kwargs = {"train": True},
        test_kwargs  = {"train": False},
        check_path   = "FashionMNIST/raw/train-images-idx3-ubyte",
    ),

    "kmnist": DatasetInfo(
        key          = "kmnist",
        display_name = "Kuzushiji-MNIST",
        cls          = KMNISTDataset,       # HuggingFace-backed; see core/data/kmnist.py
        train_kwargs = {"train": True},
        test_kwargs  = {"train": False},
        check_path   = "KMNIST/raw/train-images-idx3-ubyte",
    ),

    "omniglot": DatasetInfo(
        key          = "omniglot",
        display_name = "Omniglot",
        cls          = tv_datasets.Omniglot,
        train_kwargs = {"background": True},
        test_kwargs  = {"background": False},
        check_path   = "omniglot-py/images_background",
    ),

    "not_mnist": DatasetInfo(
        key          = "not_mnist",
        display_name = "notMNIST",
        cls          = NotMNISTDataset,     # HuggingFace-backed; see core/data/notmnist.py
        train_kwargs = {"train": True},
        test_kwargs  = {"train": False},
        check_path   = "notMNIST/raw/train-images.npy",
    ),
}


# ── Download-status helpers ────────────────────────────────────────────────────

def is_downloaded(name: str) -> bool:
    """
    Fast filesystem check — does not verify file integrity, only existence.
    """
    info = TRAIN_DATASETS.get(name)
    if info is None:
        return False
    return (DATASETS_DIR / info.check_path).exists()


def download_dataset(name: str) -> None:
    """
    Trigger a download for *name* using the appropriate class.

    - KMNIST / notMNIST: downloaded via HuggingFace Hub and cached locally.
    - All other datasets: delegated to torchvision's built-in download.
    """
    info = TRAIN_DATASETS.get(name)
    if info is None:
        raise ValueError(f"Unknown dataset '{name}'.")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    transform = _get_transform(name)

    # kmnist and not_mnist expect a Path root; torchvision classes expect str.
    root = DATASETS_DIR if name in ("kmnist", "not_mnist") else str(DATASETS_DIR)

    info.cls(
        root      = root,
        download  = True,
        transform = transform,
        **info.train_kwargs,
    )


# ── Transform selection ────────────────────────────────────────────────────────

def _get_transform(name: str) -> Callable:
    """Return the appropriate eval/download-time transform for *name*."""
    if name == "omniglot":
        return get_omniglot_transform()
    if name == "not_mnist":
        return get_notmnist_transform()
    return get_mnist_transform()


# ── DataManager ────────────────────────────────────────────────────────────────

class DataManager:
    """
    Centralised dataset and DataLoader factory.

    The training dataset is determined by config.training.train_dataset.
    OOD loaders are built for all datasets listed in
    config.inference.ood_datasets that are either downloaded or network-backed,
    excluding the training dataset itself.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        augment_train: bool = False,
    ) -> None:
        self.config        = config
        self.augment_train = augment_train
        self.root          = DATASETS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

        self._train_loader: DataLoader | None = None
        self._val_loader:   DataLoader | None = None
        self._test_loader:  DataLoader | None = None
        self._ood_loaders:  dict[str, DataLoader] | None = None

        self._generator = torch.Generator().manual_seed(_SPLIT_SEED)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_train_loader(self) -> DataLoader:
        if self._train_loader is None:
            self._build_train_val_loaders()
        return self._train_loader   # type: ignore[return-value]

    def get_val_loader(self) -> DataLoader:
        if self._val_loader is None:
            self._build_train_val_loaders()
        return self._val_loader     # type: ignore[return-value]

    def get_test_loader(self) -> DataLoader:
        if self._test_loader is None:
            self._build_test_loader()
        return self._test_loader    # type: ignore[return-value]

    def get_ood_loaders(self) -> dict[str, DataLoader]:
        if self._ood_loaders is None:
            self._build_ood_loaders()
        return self._ood_loaders    # type: ignore[return-value]

    def get_ood_loader(self, name: str) -> DataLoader:
        return self.get_ood_loaders()[name]

    # ── Train / val split ──────────────────────────────────────────────────────

    def _build_train_val_loaders(self) -> None:
        name  = self.config.training.train_dataset
        info  = TRAIN_DATASETS[name]
        t_cfg = self.config.training

        if name == "mnist" and self.augment_train:
            from core.data.transforms import get_augmented_mnist_transform
            train_transform = get_augmented_mnist_transform()
        else:
            train_transform = _get_transform(name)

        full_train = self._make_dataset(name, info, train_transform, split="train")

        if t_cfg.train_size < 1.0:
            n_keep  = max(1, int(len(full_train) * t_cfg.train_size))
            indices = torch.randperm(
                len(full_train), generator=self._generator
            )[:n_keep].tolist()
            full_train = Subset(full_train, indices)

        n_total = len(full_train)
        n_val   = max(1, int(n_total * t_cfg.val_split))
        n_train = n_total - n_val

        train_subset, val_subset = random_split(
            full_train, [n_train, n_val], generator=self._generator,
        )
        self._train_loader = self._make_loader(train_subset, shuffle=True)
        self._val_loader   = self._make_loader(val_subset,   shuffle=False)

    # ── Test loader ────────────────────────────────────────────────────────────

    def _build_test_loader(self) -> None:
        name  = self.config.training.train_dataset
        info  = TRAIN_DATASETS[name]
        test_set = self._make_dataset(name, info, _get_transform(name), split="test")
        self._test_loader = self._make_loader(test_set, shuffle=False)

    # ── OOD loaders ────────────────────────────────────────────────────────────

    def _build_ood_loaders(self) -> None:
        train_name = self.config.training.train_dataset
        loaders: dict[str, DataLoader] = {}

        for name in self.config.inference.ood_datasets:
            if name == train_name:
                continue

            info = TRAIN_DATASETS.get(name)
            if info is None:
                print(f"[DataManager] Unknown OOD dataset '{name}', skipping.")
                continue

            if not is_downloaded(name):
                print(f"[DataManager] OOD dataset '{name}' not downloaded, skipping.")
                continue

            try:
                dataset = self._make_dataset(name, info, _get_transform(name), split="test")
                loaders[name] = self._make_loader(dataset, shuffle=False)
                print(
                    f"[DataManager] OOD '{info.display_name}': "
                    f"{len(dataset)} samples."   # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"[DataManager] Warning: skipping '{name}': {exc}")

        self._ood_loaders = loaders

    # ── Dataset factory ────────────────────────────────────────────────────────

    def _make_dataset(
        self,
        name: str,
        info: DatasetInfo,
        transform: Callable,
        split: str,         # "train" or "test"
    ) -> Dataset:
        """Instantiate a dataset, routing root type and kwargs by split."""
        kwargs = info.train_kwargs if split == "train" else info.test_kwargs

        # KMNISTDataset and NotMNISTDataset expect a Path; torchvision classes expect str.
        root = self.root if name in ("kmnist", "not_mnist") else str(self.root)

        # Always allow download so torchvision can fetch any missing split
        # (e.g. Omniglot needs both images_background and images_evaluation).
        # Custom datasets (kmnist, not_mnist) handle cache-hit checks themselves;
        # torchvision datasets skip the network if files are already present.
        return info.cls(
            root      = root,
            download  = True,
            transform = transform,
            **kwargs,
        )

    # ── Loader factory ─────────────────────────────────────────────────────────

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size         = self.config.training.batch_size,
            shuffle            = shuffle,
            num_workers        = self._num_workers(),
            pin_memory         = (self.config.device == "cuda"),
            persistent_workers = False,
        )

    @staticmethod
    def _num_workers() -> int:
        import sys, os
        return 0 if sys.platform == "win32" \
            else min(4, (os.cpu_count() or 1) // 2)

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, int]:
        sizes = {
            "train": len(self.get_train_loader().dataset),  # type: ignore
            "val":   len(self.get_val_loader().dataset),    # type: ignore
            "test":  len(self.get_test_loader().dataset),   # type: ignore
        }
        for name, loader in self.get_ood_loaders().items():
            sizes[f"ood_{name}"] = len(loader.dataset)     # type: ignore
        return sizes

    def __repr__(self) -> str:
        return (
            f"DataManager("
            f"train='{self.config.training.train_dataset}', "
            f"batch={self.config.training.batch_size})"
        )