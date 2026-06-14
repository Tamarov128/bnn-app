"""
app/workers/inference_worker.py
────────────────────────────────
QThread worker that runs Evaluator.run() off the main thread.

Used in two contexts:
  1. TestingTab  — full evaluation run over test + OOD loaders.
  2. DrawingTab  — single-sample inference (fast enough to be near-synchronous,
                   but wrapped here for API consistency and to avoid any
                   chance of blocking the Qt event loop during T=50 passes).

Signal contract
───────────────
  result_ready(EvalResults)
      Emitted once when a full evaluation run completes successfully.

  single_result_ready(PredictionResult)
      Emitted once when a single-sample inference call completes.
      Used by the DrawingTab to update the live probability bar chart.

  progress_updated(step: int, total: int)
      Emitted after each dataset is evaluated; drives a QProgressBar.

  status_updated(message: str)
      Free-text progress messages forwarded from the Evaluator.

  error_occurred(message: str)
      Full traceback on any unhandled exception inside run().
"""

from __future__ import annotations

import traceback
from enum import Enum, auto
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from PyQt6.QtCore import QThread, pyqtSignal

from core.config import ExperimentConfig
from core.evaluation.evaluator import EvalResults, Evaluator
from core.inference.base import PredictionResult
from core.inference.deterministic import DeterministicPredictor
from core.inference.mc_dropout import MCDropoutPredictor
from core.models.alexnet import AlexNetSmall


class _SingleMCResult:
    """
    Lightweight stand-in for PredictionResult used by _run_single's MC branch.

    Carries only the fields DrawingTab._on_worker_result actually reads:
      result.probs[0]   → (C,) mean class probabilities
      result.entropy[0] → scalar predictive entropy
      result.mi[0]      → scalar mutual information (epistemic uncertainty)
    """
    __slots__ = ("probs", "entropy", "mi")

    def __init__(
        self,
        probs:   np.ndarray,
        entropy: np.ndarray,
        mi:      np.ndarray,
    ) -> None:
        self.probs   = probs
        self.entropy = entropy
        self.mi      = mi


class InferenceMode(Enum):
    DETERMINISTIC = auto()
    MC_DROPOUT    = auto()
    BOTH          = auto()        # runs det. then MC; emits two result_ready


class InferenceWorker(QThread):
    """
    Background thread for evaluation and single-sample inference.

    Parameters
    ----------
    config      : ExperimentConfig
    model       : LeNet5
        Pre-loaded, weights already applied.
    test_loader : DataLoader, optional
        Required for full evaluation mode.
    ood_loaders : dict[str, DataLoader], optional
        Required for full evaluation mode.
    single_image : torch.Tensor, optional
        Shape (1, 1, 28, 28).  When set, the worker runs single-sample
        inference instead of a full evaluation loop.
    mode : InferenceMode
        Which predictor(s) to run.  Defaults to BOTH.
    """

    # ── Signals ────────────────────────────────────────────────────────────────

    result_ready        = pyqtSignal(object)   # EvalResults
    single_result_ready = pyqtSignal(object)   # PredictionResult
    progress_updated    = pyqtSignal(int, int)
    status_updated      = pyqtSignal(str)
    error_occurred      = pyqtSignal(str)

    # ── Init ───────────────────────────────────────────────────────────────────

    def __init__(
        self,
        config:       ExperimentConfig,
        model:        AlexNetSmall,
        test_loader:  Optional[DataLoader]       = None,
        ood_loaders:  Optional[dict]             = None,
        single_image: Optional[torch.Tensor]     = None,
        mode:         InferenceMode              = InferenceMode.BOTH,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config       = config
        self.model        = model
        self.test_loader  = test_loader
        self.ood_loaders  = ood_loaders or {}
        self.single_image = single_image
        self.mode         = mode

    # ── Thread entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            if self.single_image is not None:
                self._run_single()
            else:
                self._run_evaluation()
        except Exception:
            self.error_occurred.emit(traceback.format_exc())

    # ── Single-sample inference ────────────────────────────────────────────────

    def _run_single(self) -> None:
        """
        Run inference on self.single_image and emit single_result_ready.

        When mode is BOTH, two signals are emitted in order:
          1. deterministic result
          2. MC Dropout result (T forward passes with dropout active)
        DrawingTab._on_worker_result uses _det_result=None as a sentinel
        to distinguish them, so order must be preserved.
        """
        from torch.utils.data import TensorDataset

        image  = self.single_image                        # (1, C, H, W)
        labels = torch.zeros(1, dtype=torch.long)         # dummy label
        loader = DataLoader(
            TensorDataset(image, labels), batch_size=1
        )

        # ── Deterministic pass ────────────────────────────────────────────────
        if self.mode in (InferenceMode.DETERMINISTIC, InferenceMode.BOTH):
            predictor = DeterministicPredictor(self.model, self.config)
            result    = predictor.predict(loader)
            self.single_result_ready.emit(result)

        # ── MC Dropout pass ───────────────────────────────────────────────────
        if self.mode in (InferenceMode.MC_DROPOUT, InferenceMode.BOTH):
            T      = self.config.inference.mc_samples if self.config else 50
            device = next(self.model.parameters()).device
            tensor = image.to(device)

            self.model.eval()
            self.model.enable_dropout()          # re-activate dropout layers

            with torch.no_grad():
                passes = torch.stack([           # (T, 1, C)
                    torch.softmax(self.model(tensor), dim=1)
                    for _ in range(T)
                ]).squeeze(1).cpu().numpy()      # (T, C)

            mc_probs     = passes.mean(axis=0)                        # (C,)
            mc_ent       = float(-np.sum(
                np.clip(mc_probs, 1e-10, 1.0) * np.log(np.clip(mc_probs, 1e-10, 1.0))
            ))
            mean_pp_ent  = float(np.mean([
                -np.sum(np.clip(p, 1e-10, 1.0) * np.log(np.clip(p, 1e-10, 1.0)))
                for p in passes
            ]))
            mc_mi        = float(max(mc_ent - mean_pp_ent, 0.0))

            # Build a PredictionResult-compatible object so the signal contract
            # stays identical to the deterministic branch.
            result = _SingleMCResult(
                probs   = mc_probs[np.newaxis],  # (1, C) — matches result.probs[0]
                entropy = np.array([mc_ent]),     # (1,)
                mi      = np.array([mc_mi]),      # (1,)
            )
            self.single_result_ready.emit(result)

    # ── Full evaluation ────────────────────────────────────────────────────────

    def _run_evaluation(self) -> None:
        if self.test_loader is None:
            raise ValueError("test_loader is required for full evaluation mode.")

        modes_to_run = []
        if self.mode in (InferenceMode.DETERMINISTIC, InferenceMode.BOTH):
            modes_to_run.append(("Deterministic", DeterministicPredictor))
        if self.mode in (InferenceMode.MC_DROPOUT, InferenceMode.BOTH):
            modes_to_run.append((
                f"MC Dropout (T={self.config.inference.mc_samples})",
                MCDropoutPredictor,
            ))

        for pred_name, PredClass in modes_to_run:
            self.status_updated.emit(f"Evaluating {pred_name}…")
            predictor = PredClass(self.model, self.config)

            evaluator = Evaluator(
                predictor      = predictor,
                predictor_name = pred_name,
                on_progress    = self.progress_updated.emit,
            )
            eval_result = evaluator.run(self.test_loader, self.ood_loaders)
            self.result_ready.emit(eval_result)

        self.status_updated.emit("Evaluation complete.")