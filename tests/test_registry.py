"""
tests/test_registry.py
────────────────────────
Tests for ModelRegistry (registry.py) and GalleryRegistry (gallery_registry.py).

All tests use a fresh temporary directory so they are hermetic and can run
in any order without sharing state.

Coverage
────────
ModelRegistry
  • save_model / list_models round-trip
  • load_state_dict correctness
  • update_eval_metrics persistence
  • delete_model removes .pt file and registry entry
  • name uniqueness (_unique_name suffix logic)
  • atomic write: registry.json updated only via .tmp rename
  • corrupt registry.json → backup created, fresh registry started
  • missing .pt file → FileNotFoundError

GalleryRegistry
  • save / load_for_model round-trip
  • image PNG is persisted on disk
  • load_for_model returns newest-first order
  • delete removes record and PNG
  • corrupt registry.json → backup and empty list
  • GalleryRecord.to_dict / from_dict are inverses
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ══════════════════════════════════════════════════════════════════════════════
# Minimal stubs shared by both registry test classes
# ══════════════════════════════════════════════════════════════════════════════

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _make_state_dict() -> dict:
    return TinyModel().state_dict()


# Minimal ExperimentConfig stub
@dataclass
class _TrainingCfg:
    train_dataset: str   = "mnist"
    batch_size:    int   = 32
    val_split:     float = 0.1
    train_size:    float = 1.0
    epochs:        int   = 5
    lr:            float = 1e-3


@dataclass
class _InferenceCfg:
    mc_samples:   int       = 10
    ood_datasets: list[str] = field(default_factory=list)


@dataclass
class FakeConfig:
    device:    str           = "cpu"
    training:  _TrainingCfg  = field(default_factory=_TrainingCfg)
    inference: _InferenceCfg = field(default_factory=_InferenceCfg)

    def to_dict(self) -> dict:
        return {
            "device":    self.device,
            "training":  self.training.__dict__.copy(),
            "inference": {
                "mc_samples":   self.inference.mc_samples,
                "ood_datasets": list(self.inference.ood_datasets),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FakeConfig":
        tr  = _TrainingCfg(**d.get("training",  {}))
        inf = _InferenceCfg(**d.get("inference", {}))
        return cls(device=d.get("device", "cpu"), training=tr, inference=inf)


# ══════════════════════════════════════════════════════════════════════════════
# Fixture helpers
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def tmp_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "saved_models"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_gallery_dir(tmp_path: Path) -> Path:
    d = tmp_path / "exports" / "gallery"
    d.mkdir(parents=True)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# ModelRegistry — imported with patched SAVED_MODELS_DIR
# ══════════════════════════════════════════════════════════════════════════════

def _make_registry(models_dir: Path):
    """Import ModelRegistry with SAVED_MODELS_DIR redirected to *models_dir*."""
    import importlib
    import core.registry as reg_mod

    with patch.object(reg_mod, "SAVED_MODELS_DIR", models_dir):
        # Re-instantiate so it picks up the patched dir
        instance = object.__new__(reg_mod.ModelRegistry)
        reg_mod.ModelRegistry._REGISTRY_FILE.__set__ = None  # type: ignore
        # Manually set the registry file path
        instance.__dict__["_entries"] = []
        # Call __init__ manually with patched constant
        with patch("core.registry.SAVED_MODELS_DIR", models_dir):
            return reg_mod.ModelRegistry()


class TestModelRegistry:
    """Tests for ModelRegistry save/load/delete and edge cases."""

    @pytest.fixture(autouse=True)
    def registry(self, tmp_models_dir):
        """Fresh ModelRegistry backed by a temp directory."""
        import core.registry as reg_mod
        with patch("core.registry.SAVED_MODELS_DIR", tmp_models_dir):
            reg_mod.ModelRegistry._REGISTRY_FILE = tmp_models_dir / "registry.json"
            self.reg = reg_mod.ModelRegistry()
            self.reg_mod = reg_mod
            self.models_dir = tmp_models_dir
            yield

    def test_initially_empty(self):
        assert len(self.reg) == 0
        assert self.reg.list_models() == []

    # ── save and list ─────────────────────────────────────────────────────────

    def test_save_creates_pt_file(self):
        sd = _make_state_dict()
        entry = self.reg.save_model("model_a", sd, FakeConfig())
        assert entry.path.exists()

    def test_save_adds_to_list(self):
        self.reg.save_model("model_a", _make_state_dict(), FakeConfig())
        assert len(self.reg) == 1

    def test_list_models_newest_first(self):
        self.reg.save_model("first",  _make_state_dict(), FakeConfig())
        self.reg.save_model("second", _make_state_dict(), FakeConfig())
        names = [e.name for e in self.reg.list_models()]
        assert names[0] == "second"
        assert names[1] == "first"

    # ── round-trip ────────────────────────────────────────────────────────────

    def test_state_dict_round_trip(self):
        original_sd = _make_state_dict()
        entry = self.reg.save_model("rt_model", original_sd, FakeConfig())
        loaded_sd = self.reg.load_state_dict(entry)
        for key in original_sd:
            torch.testing.assert_close(original_sd[key], loaded_sd[key])

    def test_config_round_trip(self):
        cfg   = FakeConfig(device="cpu")
        entry = self.reg.save_model("cfg_test", _make_state_dict(), cfg)
        # ModelEntry stores config as dict; we can reconstruct via from_dict
        restored = FakeConfig.from_dict(entry.config)
        assert restored.device == cfg.device
        assert restored.training.train_dataset == cfg.training.train_dataset

    # ── update_eval_metrics ───────────────────────────────────────────────────

    def test_update_eval_metrics(self):
        entry = self.reg.save_model("eval_test", _make_state_dict(), FakeConfig())
        metrics = {"auroc": 0.95, "fpr95": 0.12}
        self.reg.update_eval_metrics(entry.name, metrics)

        updated = self.reg.get_entry(entry.name)
        assert updated.eval_metrics == metrics

    def test_update_eval_metrics_persisted_to_disk(self):
        """After update the JSON on disk should reflect the new metrics."""
        entry = self.reg.save_model("persist_test", _make_state_dict(), FakeConfig())
        self.reg.update_eval_metrics(entry.name, {"accuracy": 0.99})

        raw = json.loads((self.models_dir / "registry.json").read_text())
        stored = next(r for r in raw if r["name"] == "persist_test")
        assert stored["eval_metrics"]["accuracy"] == pytest.approx(0.99)

    def test_update_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.reg.update_eval_metrics("ghost_model", {})

    # ── delete ────────────────────────────────────────────────────────────────

    def test_delete_removes_pt_file(self):
        entry = self.reg.save_model("del_model", _make_state_dict(), FakeConfig())
        pt_path = entry.path
        self.reg.delete_model(entry.name)
        assert not pt_path.exists()

    def test_delete_removes_registry_entry(self):
        entry = self.reg.save_model("del_model2", _make_state_dict(), FakeConfig())
        self.reg.delete_model(entry.name)
        assert len(self.reg) == 0
        assert self.reg.get_entry(entry.name) is None

    def test_delete_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.reg.delete_model("nothing_here")

    # ── name uniqueness ───────────────────────────────────────────────────────

    def test_duplicate_name_gets_suffix(self):
        self.reg.save_model("dup", _make_state_dict(), FakeConfig())
        entry2 = self.reg.save_model("dup", _make_state_dict(), FakeConfig())
        assert entry2.name == "dup_2"

    def test_triple_duplicate(self):
        self.reg.save_model("dup", _make_state_dict(), FakeConfig())
        self.reg.save_model("dup", _make_state_dict(), FakeConfig())
        entry3 = self.reg.save_model("dup", _make_state_dict(), FakeConfig())
        assert entry3.name == "dup_3"

    # ── atomic write ─────────────────────────────────────────────────────────

    def test_atomic_write_uses_tmp_then_rename(self):
        """
        After save_model completes, registry.json must exist and the .tmp
        file must NOT be left behind.
        """
        self.reg.save_model("atomic_test", _make_state_dict(), FakeConfig())
        registry_file = self.models_dir / "registry.json"
        tmp_file      = self.models_dir / "registry.json.tmp"
        assert registry_file.exists()
        assert not tmp_file.exists()

    def test_registry_json_is_valid_json(self):
        self.reg.save_model("json_test", _make_state_dict(), FakeConfig())
        raw = (self.models_dir / "registry.json").read_text()
        parsed = json.loads(raw)  # must not raise
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    # ── corrupt registry recovery ────────────────────────────────────────────

    def test_corrupt_registry_creates_backup_and_fresh_start(self):
        # Write a corrupt registry
        registry_file = self.models_dir / "registry.json"
        registry_file.write_text("this is not json {{{", encoding="utf-8")

        import core.registry as reg_mod
        with patch("core.registry.SAVED_MODELS_DIR", self.models_dir):
            reg_mod.ModelRegistry._REGISTRY_FILE = registry_file
            fresh = reg_mod.ModelRegistry()

        assert len(fresh) == 0
        assert (self.models_dir / "registry.json.bak").exists()

    # ── missing .pt file ─────────────────────────────────────────────────────

    def test_load_missing_pt_raises(self):
        entry = self.reg.save_model("missing_pt", _make_state_dict(), FakeConfig())
        entry.path.unlink()          # delete the .pt file manually
        with pytest.raises(FileNotFoundError):
            self.reg.load_state_dict(entry)

    # ── ModelEntry repr ───────────────────────────────────────────────────────

    def test_model_entry_repr(self):
        entry = self.reg.save_model("repr_test", _make_state_dict(), FakeConfig())
        r = repr(entry)
        assert "repr_test" in r
        assert "repr_test.pt" in r


# ══════════════════════════════════════════════════════════════════════════════
# GalleryRegistry
# ══════════════════════════════════════════════════════════════════════════════

def _make_inference_record(pred_class: int = 3):
    """Return a minimal InferenceRecord-compatible dict."""
    probs = [0.0] * 10
    probs[pred_class] = 1.0
    return {"probs": probs, "entropy": 0.05, "pred_class": pred_class}


def _random_image_28() -> np.ndarray:
    return np.random.randint(0, 256, (28, 28), dtype=np.uint8)


class TestGalleryRegistry:
    """Tests for GalleryRegistry save/load/delete and edge cases."""

    @pytest.fixture(autouse=True)
    def registry(self, tmp_gallery_dir):
        import core.gallery_registry as gal_mod
        # Patch the module-level constants
        with patch("core.gallery_registry._GALLERY_DIR",   tmp_gallery_dir), \
             patch("core.gallery_registry._REGISTRY_FILE", tmp_gallery_dir / "registry.json"):
            # Reimport to pick up patched values
            self.gal_mod     = gal_mod
            self.gallery_dir = tmp_gallery_dir
            self.reg_file    = tmp_gallery_dir / "registry.json"
            gal_mod._GALLERY_DIR   = tmp_gallery_dir
            gal_mod._REGISTRY_FILE = self.reg_file
            self.reg = gal_mod.GalleryRegistry()
            yield

    def _det(self, cls=3):
        return self.gal_mod.InferenceRecord(
            probs=[0.0] * 10, entropy=0.05, pred_class=cls
        )

    def _mc(self, cls=3):
        return self.gal_mod.InferenceRecord(
            probs=[0.0] * 10, entropy=0.12, pred_class=cls
        )

    # ── save / load round-trip ────────────────────────────────────────────────

    def test_save_creates_png(self):
        record = self.reg.save("model_x", _random_image_28(), self._det(), self._mc())
        assert record.image_path.exists()

    def test_save_png_is_28x28(self):
        from PIL import Image as PILImage
        record = self.reg.save("model_x", _random_image_28(), self._det(), self._mc())
        img = PILImage.open(record.image_path)
        assert img.size == (28, 28)

    def test_save_adds_entry(self):
        self.reg.save("model_x", _random_image_28(), self._det(), self._mc())
        assert len(self.reg.all_records()) == 1

    def test_load_for_model_returns_correct_entries(self):
        self.reg.save("model_a", _random_image_28(), self._det(), self._mc())
        self.reg.save("model_b", _random_image_28(), self._det(), self._mc())
        entries_a = self.reg.load_for_model("model_a")
        entries_b = self.reg.load_for_model("model_b")
        assert len(entries_a) == 1
        assert len(entries_b) == 1
        assert entries_a[0].model_name == "model_a"

    def test_load_for_model_newest_first(self):
        """Three saves for the same model → newest entry appears first."""
        for i in range(3):
            self.reg.save("model_z", _random_image_28(), self._det(i), self._mc(i))
        entries = self.reg.load_for_model("model_z")
        assert len(entries) == 3
        # newest is last-saved → check entry_ids are in reverse-save order
        ids = [e.entry_id for e in entries]
        assert ids == sorted(ids, reverse=True)

    def test_load_for_unknown_model_returns_empty(self):
        assert self.reg.load_for_model("no_such_model") == []

    # ── inference record fields ───────────────────────────────────────────────

    def test_det_probs_array_shape(self):
        record = self.reg.save("m", _random_image_28(), self._det(5), self._mc(5))
        assert record.det_probs_array.shape == (10,)

    def test_mc_probs_array_shape(self):
        record = self.reg.save("m", _random_image_28(), self._det(), self._mc())
        assert record.mc_probs_array.shape == (10,)

    def test_pred_class_preserved(self):
        det = self._det(cls=7)
        mc  = self._mc(cls=2)
        record = self.reg.save("model_y", _random_image_28(), det, mc)
        reloaded = self.reg.load_for_model("model_y")[0]
        assert reloaded.det.pred_class == 7
        assert reloaded.mc.pred_class  == 2

    # ── delete ────────────────────────────────────────────────────────────────

    def test_delete_removes_png(self):
        record = self.reg.save("model_del", _random_image_28(), self._det(), self._mc())
        png_path = record.image_path
        self.reg.delete(record.entry_id)
        assert not png_path.exists()

    def test_delete_removes_registry_entry(self):
        record = self.reg.save("model_del2", _random_image_28(), self._det(), self._mc())
        self.reg.delete(record.entry_id)
        assert len(self.reg.all_records()) == 0

    def test_delete_nonexistent_is_silent(self):
        """Deleting an ID that doesn't exist should not raise."""
        self.reg.delete("does_not_exist_id")  # must not raise

    def test_delete_only_removes_target(self):
        r1 = self.reg.save("m", _random_image_28(), self._det(), self._mc())
        r2 = self.reg.save("m", _random_image_28(), self._det(), self._mc())
        self.reg.delete(r1.entry_id)
        remaining = self.reg.all_records()
        assert len(remaining) == 1
        assert remaining[0].entry_id == r2.entry_id

    # ── atomic write ─────────────────────────────────────────────────────────

    def test_atomic_write_no_tmp_leftover(self):
        self.reg.save("m", _random_image_28(), self._det(), self._mc())
        tmp = self.gallery_dir / "registry.json.tmp"
        assert not tmp.exists()

    def test_registry_json_valid_after_save(self):
        self.reg.save("m", _random_image_28(), self._det(), self._mc())
        raw = json.loads(self.reg_file.read_text())
        assert isinstance(raw, list)
        assert len(raw) == 1

    # ── corrupt registry recovery ────────────────────────────────────────────

    def test_corrupt_registry_creates_backup(self):
        self.reg_file.write_text("not json at all {{{{", encoding="utf-8")
        import core.gallery_registry as gal_mod
        gal_mod._GALLERY_DIR   = self.gallery_dir
        gal_mod._REGISTRY_FILE = self.reg_file
        fresh = gal_mod.GalleryRegistry()
        assert fresh.all_records() == []
        assert (self.gallery_dir / "registry.json.bak").exists()

    # ── GalleryRecord serialisation ───────────────────────────────────────────

    def test_to_dict_from_dict_round_trip(self):
        record   = self.reg.save("serial", _random_image_28(), self._det(4), self._mc(9))
        d        = record.to_dict()
        restored = self.gal_mod.GalleryRecord.from_dict(d)
        assert restored.entry_id         == record.entry_id
        assert restored.model_name       == record.model_name
        assert restored.det.pred_class   == record.det.pred_class
        assert restored.mc.pred_class    == record.mc.pred_class
        assert restored.image_file       == record.image_file

    def test_to_dict_is_json_serialisable(self):
        record = self.reg.save("json_serial", _random_image_28(), self._det(), self._mc())
        d      = record.to_dict()
        json.dumps(d)   # must not raise

    def test_image_path_property(self):
        record = self.reg.save("path_test", _random_image_28(), self._det(), self._mc())
        assert record.image_path == self.gallery_dir / record.image_file

    # ── persistence across reload ─────────────────────────────────────────────

    def test_records_survive_reload(self):
        """Entries saved in one GalleryRegistry instance are visible in a new one."""
        self.reg.save("model_persist", _random_image_28(), self._det(1), self._mc(2))

        import core.gallery_registry as gal_mod
        gal_mod._GALLERY_DIR   = self.gallery_dir
        gal_mod._REGISTRY_FILE = self.reg_file
        new_reg = gal_mod.GalleryRegistry()

        entries = new_reg.load_for_model("model_persist")
        assert len(entries) == 1
        assert entries[0].det.pred_class == 1
        assert entries[0].mc.pred_class  == 2