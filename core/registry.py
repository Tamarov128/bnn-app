"""
core/registry.py
─────────────────
ModelRegistry: persistence layer for trained models.

Each saved model consists of:
  1. A .pt file containing the model state_dict.
  2. An entry in registry.json recording metadata.

The registry.json lives at saved_models/registry.json and is read entirely
into memory on construction; writes flush the full file atomically.

ModelEntry
──────────
Dataclass representing one row in the registry.  All fields are
JSON-serialisable so the registry file is human-readable.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from core.config import ExperimentConfig, SAVED_MODELS_DIR


# ── ModelEntry ─────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """
    Metadata record for a single saved model.

    Attributes
    ----------
    name : str
        Human-readable name chosen by the user at save time.
    filename : str
        Basename of the .pt file, e.g. "baseline_v1.pt".
    config : dict
        ExperimentConfig serialised as a dict (JSON round-trippable).
    timestamp : str
        ISO-8601 UTC timestamp of when the model was saved.
    training_metrics : dict
        Summary from TrainingResult.last_metrics() — loss/accuracy history.
    eval_metrics : dict
        Summary from EvalResults.to_dict() — filled in after evaluation,
        empty dict if the model has not been evaluated yet.
    """
    name:             str
    filename:         str
    config:           dict
    timestamp:        str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    training_metrics: dict  = field(default_factory=dict)
    eval_metrics:     dict  = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """Absolute path to the .pt file."""
        return SAVED_MODELS_DIR / self.filename

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelEntry":
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"ModelEntry(name='{self.name}', "
            f"file='{self.filename}', "
            f"saved={self.timestamp[:10]})"
        )


# ── ModelRegistry ──────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Manages the saved_models directory and registry.json manifest.

    All public methods keep the in-memory list and the JSON file in sync.
    The registry file is written atomically (write to .tmp, then rename)
    to avoid corruption on crash.

    Usage
    ─────
    >>> registry = ModelRegistry()
    >>> entry = registry.save_model(
    ...     name="baseline_v1",
    ...     state_dict=model.state_dict(),
    ...     config=cfg,
    ...     training_metrics=result.last_metrics(),
    ... )
    >>> entries = registry.list_models()
    >>> model.load_state_dict(registry.load_state_dict(entries[0]))
    """

    _REGISTRY_FILE = SAVED_MODELS_DIR / "registry.json"

    def __init__(self) -> None:
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._entries: list[ModelEntry] = self._load_registry()

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_model(
        self,
        name: str,
        state_dict: dict,
        config: ExperimentConfig,
        training_metrics: Optional[dict] = None,
    ) -> ModelEntry:
        """
        Persist a trained model and add it to the registry.

        Parameters
        ----------
        name : str
            User-chosen display name.  A numeric suffix is appended
            automatically if the name already exists.
        state_dict : dict
            Output of model.state_dict() (weights on CPU).
        config : ExperimentConfig
        training_metrics : dict, optional
            Summary dict from TrainingResult.last_metrics().

        Returns
        -------
        ModelEntry
            The newly created registry entry.
        """
        safe_name = self._unique_name(name)
        filename  = f"{safe_name}.pt"
        pt_path   = SAVED_MODELS_DIR / filename

        # Save weights
        torch.save(state_dict, pt_path)

        entry = ModelEntry(
            name             = safe_name,
            filename         = filename,
            config           = config.to_dict(),
            training_metrics = training_metrics or {},
        )
        self._entries.append(entry)
        self._flush()
        return entry

    def update_eval_metrics(self, name: str, eval_metrics: dict) -> None:
        """
        Attach evaluation results to an existing registry entry.

        Parameters
        ----------
        name : str
            Exact entry name as stored in the registry.
        eval_metrics : dict
            Output of EvalResults.to_dict().
        """
        entry = self._get_entry(name)
        if entry is None:
            raise KeyError(f"No model named '{name}' in registry.")
        entry.eval_metrics = eval_metrics
        self._flush()

    def list_models(self) -> list[ModelEntry]:
        """Return all registry entries, newest first."""
        return list(reversed(self._entries))

    def load_state_dict(self, entry: ModelEntry) -> dict:
        """
        Load and return the state_dict for *entry*.

        Parameters
        ----------
        entry : ModelEntry

        Returns
        -------
        dict
            state_dict suitable for model.load_state_dict().
        """
        if not entry.path.exists():
            raise FileNotFoundError(
                f"Weight file not found: {entry.path}\n"
                "The registry entry may be stale."
            )
        return torch.load(entry.path, map_location="cpu", weights_only=True)

    def load_config(self, entry: ModelEntry) -> ExperimentConfig:
        """Reconstruct the ExperimentConfig from a registry entry."""
        return ExperimentConfig.from_dict(entry.config)

    def delete_model(self, name: str) -> None:
        """
        Remove a model from the registry and delete its .pt file.

        Parameters
        ----------
        name : str
            Exact entry name.
        """
        entry = self._get_entry(name)
        if entry is None:
            raise KeyError(f"No model named '{name}' in registry.")

        if entry.path.exists():
            entry.path.unlink()

        self._entries = [e for e in self._entries if e.name != name]
        self._flush()

    def get_entry(self, name: str) -> Optional[ModelEntry]:
        """Return the entry with the given name, or None if not found."""
        return self._get_entry(name)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ModelRegistry(n_models={len(self)}, path={SAVED_MODELS_DIR})"

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_registry(self) -> list[ModelEntry]:
        if not self._REGISTRY_FILE.exists():
            return []
        try:
            raw = json.loads(self._REGISTRY_FILE.read_text(encoding="utf-8"))
            return [ModelEntry.from_dict(d) for d in raw]
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupt registry — back it up and start fresh.
            backup = self._REGISTRY_FILE.with_suffix(".json.bak")
            shutil.copy(self._REGISTRY_FILE, backup)
            print(
                f"[ModelRegistry] Warning: registry.json was corrupt. "
                f"Backed up to {backup.name} and started fresh."
            )
            return []

    def _flush(self) -> None:
        """Write registry to disk atomically."""
        tmp = self._REGISTRY_FILE.with_suffix(".json.tmp")
        data = [e.to_dict() for e in self._entries]
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._REGISTRY_FILE)

    def _get_entry(self, name: str) -> Optional[ModelEntry]:
        for e in self._entries:
            if e.name == name:
                return e
        return None

    def _unique_name(self, name: str) -> str:
        """Append _2, _3, … if *name* already exists in the registry."""
        existing = {e.name for e in self._entries}
        if name not in existing:
            return name
        i = 2
        while f"{name}_{i}" in existing:
            i += 1
        return f"{name}_{i}"