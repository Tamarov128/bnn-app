"""
core/gallery_registry.py
─────────────────────────
GalleryRegistry: persists hand-drawn samples and their dual inference
results (deterministic + MC Dropout) across application sessions.

Storage layout
──────────────
    exports/
        gallery/
            registry.json          ← manifest of all saved entries
            drawing_<id>.png       ← one PNG per saved drawing (28x28)

Each registry entry stores:
  - Unique ID (timestamp string)
  - Model name (links entries to a specific saved model)
  - Both inference results (probs, entropy, predicted class)
  - Filename of the PNG on disk

When the drawing tab loads a model, it calls load_entries(model_name)
to restore that model's gallery cards.  Delete removes both the JSON
record and the PNG file.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

from core.config import EXPORTS_DIR

_GALLERY_DIR      = EXPORTS_DIR / "gallery"
_REGISTRY_FILE    = _GALLERY_DIR / "registry.json"


# ── Result sub-record ──────────────────────────────────────────────────────────

@dataclass
class InferenceRecord:
    """Inference result for one mode (deterministic or MC Dropout)."""
    probs:     list[float]    # length 10 — serialisable form of np.ndarray
    entropy:   float
    pred_class: int


# ── Gallery entry ──────────────────────────────────────────────────────────────

@dataclass
class GalleryRecord:
    """
    One saved drawing + dual inference result, fully serialisable.

    Attributes
    ----------
    entry_id    : str   — timestamp string used as unique key and filename stem
    model_name  : str   — name of the model that produced these predictions
    timestamp   : str   — ISO-8601 creation time
    image_file  : str   — basename of the PNG (relative to _GALLERY_DIR)
    det         : InferenceRecord
    mc          : InferenceRecord
    """
    entry_id:   str
    model_name: str
    timestamp:  str
    image_file: str
    det:        InferenceRecord
    mc:         InferenceRecord

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "GalleryRecord":
        return cls(
            entry_id   = data["entry_id"],
            model_name = data["model_name"],
            timestamp  = data["timestamp"],
            image_file = data["image_file"],
            det        = InferenceRecord(**data["det"]),
            mc         = InferenceRecord(**data["mc"]),
        )

    @property
    def image_path(self) -> Path:
        return _GALLERY_DIR / self.image_file

    @property
    def det_probs_array(self) -> np.ndarray:
        return np.array(self.det.probs, dtype=np.float32)

    @property
    def mc_probs_array(self) -> np.ndarray:
        return np.array(self.mc.probs, dtype=np.float32)


# ── Registry ───────────────────────────────────────────────────────────────────

class GalleryRegistry:
    """
    Manages the gallery registry.json and the associated PNG files.

    Usage
    ─────
    >>> reg = GalleryRegistry()
    >>> record = reg.save(
    ...     model_name="baseline_v1",
    ...     image_28=arr,
    ...     det=InferenceRecord(...),
    ...     mc=InferenceRecord(...),
    ... )
    >>> entries = reg.load_for_model("baseline_v1")
    >>> reg.delete(record.entry_id)
    """

    def __init__(self) -> None:
        _GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        self._records: list[GalleryRecord] = self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def save(
        self,
        model_name: str,
        image_28:   np.ndarray,        # (28, 28) uint8
        det:        InferenceRecord,
        mc:         InferenceRecord,
    ) -> GalleryRecord:
        """
        Persist a drawing and its dual inference result.

        Returns the created GalleryRecord.
        """
        entry_id   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_file = f"drawing_{entry_id}.png"
        image_path = _GALLERY_DIR / image_file

        # Save PNG
        PILImage.fromarray(image_28, mode="L").save(image_path)

        record = GalleryRecord(
            entry_id   = entry_id,
            model_name = model_name,
            timestamp  = datetime.now().isoformat(),
            image_file = image_file,
            det        = det,
            mc         = mc,
        )
        self._records.append(record)
        self._flush()
        return record

    def load_for_model(self, model_name: str) -> list[GalleryRecord]:
        """Return all records for *model_name*, newest first."""
        return [
            r for r in reversed(self._records)
            if r.model_name == model_name
        ]

    def delete(self, entry_id: str) -> None:
        """Remove the record and its PNG file."""
        record = self._find(entry_id)
        if record is None:
            return

        if record.image_path.exists():
            record.image_path.unlink()

        self._records = [r for r in self._records if r.entry_id != entry_id]
        self._flush()

    def all_records(self) -> list[GalleryRecord]:
        return list(self._records)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load(self) -> list[GalleryRecord]:
        if not _REGISTRY_FILE.exists():
            return []
        try:
            data = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
            records = []
            for d in data:
                try:
                    records.append(GalleryRecord.from_dict(d))
                except Exception:
                    pass   # skip malformed entries silently
            return records
        except (json.JSONDecodeError, TypeError):
            backup = _REGISTRY_FILE.with_suffix(".json.bak")
            shutil.copy(_REGISTRY_FILE, backup)
            return []

    def _flush(self) -> None:
        tmp = _REGISTRY_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps([r.to_dict() for r in self._records], indent=2),
            encoding="utf-8",
        )
        tmp.replace(_REGISTRY_FILE)

    def _find(self, entry_id: str) -> Optional[GalleryRecord]:
        for r in self._records:
            if r.entry_id == entry_id:
                return r
        return None