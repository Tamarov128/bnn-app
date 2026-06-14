"""
core/data/kmnist.py
───────────────────────
Custom PyTorch Dataset for Kuzushiji-MNIST (KMNIST).

Why this exists
───────────────
torchvision's built-in KMNIST class hardcodes codh.rois.ac.jp as its only
download mirror.  That server is unreliable from many networks and has
frequent outages.  This class uses HuggingFace Hub (tanganke/kmnist) as the
sole download source: it is globally distributed, requires no authentication,
and is significantly faster than the original CODH server.

On first use the dataset is converted from HuggingFace parquet format to the
standard IDX binary layout and cached under <root>/KMNIST/raw/ — exactly
where torchvision expects it — so subsequent loads are instant.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset

# HuggingFace dataset identifier.
_HF_DATASET = "tanganke/kmnist"

# File stems for the two splits (matches torchvision's naming convention).
_SPLIT_STEMS = {
    True:  ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    False: ("t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"),
}


class KMNISTDataset(Dataset):
    """
    Kuzushiji-MNIST dataset backed by HuggingFace Hub.

    Parameters
    ----------
    root : str | Path
        Root directory.  IDX files are cached under ``<root>/KMNIST/raw/``.
    train : bool
        Training split (60 000 samples) if True, test split (10 000) if False.
    download : bool
        Download from HuggingFace Hub if the data is not already cached.
    transform : callable, optional
        Callable applied to each PIL image before it is returned.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root      = Path(root) / "KMNIST" / "raw"
        self.train     = train
        self.transform = transform

        if download and not self._is_cached():
            self._download_from_huggingface()

        if not self._is_cached():
            raise RuntimeError(
                "KMNIST data not found and automatic download failed.\n\n"
                "Option 1 — install the HuggingFace datasets library and retry:\n"
                "    pip install datasets\n\n"
                "Option 2 — manual download:\n"
                "    1. Visit https://github.com/rois-codh/kmnist\n"
                "    2. Download all four IDX files from the 'Data Downloads' section.\n"
                f"   3. Place them (or their .gz versions) in:  {self.root}"
            )

        self._images, self._labels = self._load_idx()

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
        return f"KMNISTDataset(split={split}, n={len(self._labels)})"

    # ── Cache check ────────────────────────────────────────────────────────────

    def _is_cached(self) -> bool:
        """True if all four IDX files are present (raw or gzipped)."""
        for stem in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                     "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"):
            if not (self.root / stem).exists() and not (self.root / (stem + ".gz")).exists():
                return False
        return True

    # ── Download ───────────────────────────────────────────────────────────────

    def _download_from_huggingface(self) -> None:
        """
        Download both splits from HuggingFace Hub and convert to IDX format.

        Writes four .gz files to self.root so the cache is valid for future
        instantiations (and compatible with torchvision's KMNIST loader).
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' library is required to download KMNIST.\n"
                "Install it with:  pip install datasets"
            ) from exc

        self.root.mkdir(parents=True, exist_ok=True)
        print(f"[KMNIST] Downloading from HuggingFace ({_HF_DATASET}) …")

        for split_name, prefix in (("train", "train"), ("test", "t10k")):
            hf_split = load_dataset(_HF_DATASET, split=split_name)
            print(f"[KMNIST] Converting {split_name} split ({len(hf_split)} samples) …")
            self._write_idx(hf_split, prefix)

        print("[KMNIST] Download and conversion complete.")

    def _write_idx(self, hf_dataset, prefix: str) -> None:
        """Convert a HuggingFace split to gzipped IDX files on disk."""
        n = len(hf_dataset)
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
            if (i + 1) % 10_000 == 0:
                print(f"\r[KMNIST]   {i + 1}/{n}", end="", flush=True)
        print()

        # IDX3 image file: magic 0x00000803 + n + rows + cols + raw bytes
        img_path = self.root / f"{prefix}-images-idx3-ubyte.gz"
        with gzip.open(img_path, "wb") as f:
            f.write(struct.pack(">IIII", 0x00000803, n, 28, 28))
            f.write(images.tobytes())

        # IDX1 label file: magic 0x00000801 + n + raw bytes
        lbl_path = self.root / f"{prefix}-labels-idx1-ubyte.gz"
        with gzip.open(lbl_path, "wb") as f:
            f.write(struct.pack(">II", 0x00000801, n))
            f.write(labels.tobytes())

    # ── IDX loading ────────────────────────────────────────────────────────────

    def _load_idx(self) -> tuple[np.ndarray, np.ndarray]:
        img_stem, lbl_stem = _SPLIT_STEMS[self.train]
        images = _read_idx_images(self._resolve(img_stem))
        labels = _read_idx_labels(self._resolve(lbl_stem))
        return images, labels

    def _resolve(self, stem: str) -> Path:
        """Return the path to a decompressed IDX file, extracting .gz if needed."""
        raw = self.root / stem
        gz  = self.root / (stem + ".gz")
        if not raw.exists():
            if not gz.exists():
                raise FileNotFoundError(f"Neither {raw} nor {gz} found.")
            print(f"[KMNIST] Extracting {gz.name} …")
            with gzip.open(gz, "rb") as fin, open(raw, "wb") as fout:
                fout.write(fin.read())
        return raw


# ── IDX binary format readers ──────────────────────────────────────────────────

def _read_idx_images(path: Path) -> np.ndarray:
    """Parse an IDX3 image file → uint8 array of shape (n, rows, cols)."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"Bad magic number {magic:#010x} in {path}; expected 0x00000803.")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Parse an IDX1 label file → uint8 array of shape (n,)."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"Bad magic number {magic:#010x} in {path}; expected 0x00000801.")
        return np.frombuffer(f.read(), dtype=np.uint8)