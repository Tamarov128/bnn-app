"""
core/config.py
───────────────────────
Central configuration dataclass for all experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Self

# ── Project paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATASETS_DIR     = _PROJECT_ROOT / "datasets"
SAVED_MODELS_DIR = _PROJECT_ROOT / "saved_models"
EXPORTS_DIR      = _PROJECT_ROOT / "exports"

# ── Supported datasets ─────────────────────────────────────────────────────────
AVAILABLE_DATASETS = ["mnist", "fashion_mnist", "kmnist", "not_mnist", "omniglot"]
OOD_DATASETS       = AVAILABLE_DATASETS   # backward-compat alias


@dataclass
class TrainingConfig:
    """All parameters that govern training, including model architecture."""

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset: str   = "mnist"

    # ── Data split ────────────────────────────────────────────────────────────
    train_size:    float = 1.0    # fraction of training set to use
    val_split:     float = 0.1   # fraction held out for validation

    # ── Optimiser ─────────────────────────────────────────────────────────────
    epochs:        int   = 20
    lr:            float = 1e-3
    batch_size:    int   = 128
    weight_decay:  float = 1e-4

    # ── Architecture ──────────────────────────────────────────────────────────
    dropout_rate:  float = 0.5   # Bernoulli p; baked into model at construction

    def __post_init__(self) -> None:
        if self.train_dataset not in AVAILABLE_DATASETS:
            raise ValueError(
                f"train_dataset '{self.train_dataset}' not in "
                f"{AVAILABLE_DATASETS}"
            )
        assert 0 < self.epochs,               "epochs must be positive"
        assert 0 < self.batch_size,           "batch_size must be positive"
        assert 0 < self.lr,                   "lr must be positive"
        assert 0.0 < self.train_size <= 1.0,  "train_size must be in (0, 1]"
        assert 0.0 < self.val_split  < 1.0,   "val_split must be in (0, 1)"
        assert 0.0 <= self.dropout_rate < 1.0, \
            "dropout_rate must be in [0, 1)"


@dataclass
class InferenceConfig:
    """Settings for the evaluation / inference phase."""
    mc_samples:   int       = 50
    ood_datasets: list[str] = field(
        default_factory=lambda: list(AVAILABLE_DATASETS)
    )

    def __post_init__(self) -> None:
        assert self.mc_samples >= 1, "mc_samples must be at least 1"


@dataclass
class ExperimentConfig:
    """
    Top-level config passed through the entire pipeline.

    Usage
    -----
    >>> cfg = ExperimentConfig()
    >>> cfg.save("saved_models/my_run.json")
    >>> cfg2 = ExperimentConfig.load("saved_models/my_run.json")
    """

    training:  TrainingConfig  = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device:    str | None      = None
    run_name:  str             = "unnamed_run"

    def __post_init__(self) -> None:
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        training_data = dict(data.get("training", {}))

        return cls(
            training  = TrainingConfig(**training_data),
            inference = InferenceConfig(**data.get("inference", {})),
            device    = data.get("device"),
            run_name  = data.get("run_name", "unnamed_run"),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def ensure_dirs(self) -> None:
        for d in (DATASETS_DIR, SAVED_MODELS_DIR, EXPORTS_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        t = self.training
        return (
            f"ExperimentConfig("
            f"run='{self.run_name}', "
            f"dataset='{t.train_dataset}', "
            f"device='{self.device}', "
            f"epochs={t.epochs}, "
            f"lr={t.lr}, "
            f"dropout={t.dropout_rate})"
        )