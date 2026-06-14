"""
core/data/notmnist.py
───────────────────────
Custom PyTorch Dataset for notMNIST.

notMNIST contains 28x28 greyscale images of the letters A-J rendered in
various fonts.  It is commonly used as an OOD benchmark against MNIST-family
models because it occupies a similar image space (28x28, greyscale, centred
single character) while being semantically distinct (letters vs. digits).

Source
──────
HuggingFace Hub: anubhavmaity/notMNIST
    train split :  15 000 samples
    test  split :   3 750 samples  (used for OOD evaluation)

Caching
───────
On first use both splits are downloaded from HuggingFace Hub and saved as
a pair of numpy arrays under <root>/notMNIST/raw/:

    train-images.npy   shape (15000, 28, 28)  uint8
    train-labels.npy   shape (15000,)          uint8
    t10k-images.npy    shape ( 3750, 28, 28)  uint8
    t10k-labels.npy    shape ( 3750,)          uint8

Subsequent instantiations load directly from these files — no network access
required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset

_HF_DATASET = "anubhavmaity/notMNIST"

# notMNIST class labels: letters A–J mapped to indices 0–9.
CLASSES = list("ABCDEFGHIJ")

# Cache file stems keyed by split flag (mirrors KMNIST / MNIST naming style).
_SPLIT_STEMS = {
    True:  ("train-images", "train-labels"),
    False: ("t10k-images",  "t10k-labels"),
}


class NotMNISTDataset(Dataset):
    """
    notMNIST dataset with local numpy cache.

    Parameters
    ----------
    root : str | Path
        Root directory.  Arrays are cached under ``<root>/notMNIST/raw/``.
    train : bool
        Training split (15 000 samples) if True, test split (3 750) if False.
    download : bool
        Download from HuggingFace Hub if the local cache is absent.
    transform : callable, optional
        Callable applied to each PIL image before it is returned.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = False,
        download: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root      = Path(root) / "notMNIST" / "raw"
        self.train     = train
        self.transform = transform

        if download and not self._is_cached():
            self._download_from_huggingface()

        if not self._is_cached():
            raise RuntimeError(
                "notMNIST data not found and automatic download failed.\n\n"
                "Option 1 — install the HuggingFace datasets library and retry:\n"
                "    pip install datasets\n\n"
                "Option 2 — manual download:\n"
                "    1. Download both splits from HuggingFace Hub:\n"
                "       https://huggingface.co/datasets/anubhavmaity/notMNIST\n"
                "    2. Convert images and labels to uint8 numpy arrays and save as:\n"
                f"      {self.root / 'train-images.npy'}  shape (15000, 28, 28)\n"
                f"      {self.root / 'train-labels.npy'}  shape (15000,)\n"
                f"      {self.root / 't10k-images.npy'}   shape (3750, 28, 28)\n"
                f"      {self.root / 't10k-labels.npy'}   shape (3750,)\n"
            )

        self._images, self._labels = self._load_cache()

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple:
        img   = PILImage.fromarray(self._images[index], mode="L")
        label = int(self._labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __repr__(self) -> str:
        split = "train" if self.train else "test"
        return f"NotMNISTDataset(split={split}, n={len(self._labels)})"

    # ── Cache check ────────────────────────────────────────────────────────────

    def _is_cached(self) -> bool:
        """True if all four .npy cache files are present."""
        for img_stem, lbl_stem in _SPLIT_STEMS.values():
            if not (self.root / f"{img_stem}.npy").exists():
                return False
            if not (self.root / f"{lbl_stem}.npy").exists():
                return False
        return True

    # ── Download ───────────────────────────────────────────────────────────────

    def _download_from_huggingface(self) -> None:
        """Download both splits from HuggingFace Hub and save as .npy files."""
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' library is required to download notMNIST.\n"
                "Install it with:  pip install datasets"
            ) from exc

        self.root.mkdir(parents=True, exist_ok=True)
        print(f"[notMNIST] Downloading from HuggingFace ({_HF_DATASET}) …")

        for split_name, (img_stem, lbl_stem) in zip(
            ("train", "test"), _SPLIT_STEMS.values()
        ):
            hf_split = load_dataset(_HF_DATASET, split=split_name)
            print(f"[notMNIST] Caching {split_name} split ({len(hf_split)} samples) …")
            self._write_cache(hf_split, img_stem, lbl_stem)

        print("[notMNIST] Download and caching complete.")

    def _write_cache(self, hf_dataset, img_stem: str, lbl_stem: str) -> None:
        """Convert a HuggingFace split to .npy files on disk."""
        n      = len(hf_dataset)
        images = np.zeros((n, 28, 28), dtype=np.uint8)
        labels = np.zeros(n, dtype=np.uint8)

        for i, item in enumerate(hf_dataset):
            img = item["image"]
            if img.mode != "L":
                img = img.convert("L")
            if img.size != (28, 28):
                img = img.resize((28, 28), PILImage.BILINEAR)
            images[i] = np.array(img, dtype=np.uint8)
            labels[i] = int(item["label"])
            if (i + 1) % 5_000 == 0:
                print(f"\r[notMNIST]   {i + 1}/{n}", end="", flush=True)
        print()

        np.save(self.root / f"{img_stem}.npy", images)
        np.save(self.root / f"{lbl_stem}.npy", labels)

    # ── Cache loading ──────────────────────────────────────────────────────────

    def _load_cache(self) -> tuple[np.ndarray, np.ndarray]:
        img_stem, lbl_stem = _SPLIT_STEMS[self.train]
        images = np.load(self.root / f"{img_stem}.npy")
        labels = np.load(self.root / f"{lbl_stem}.npy")
        return images, labels