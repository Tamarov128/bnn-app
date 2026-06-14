"""
core/inference/base.py
───────────────────────
Abstract base class and shared result container for all inference modes.

PredictionResult
────────────────
A dataclass that carries everything downstream consumers (Evaluator,
metrics modules, GUI widgets) need.  Both DeterministicPredictor and
MCDropoutPredictor return an instance of this class so all evaluation
code is inference-mode agnostic.

BasePredictor
─────────────
Defines the single required method: predict(loader) -> PredictionResult.
Concrete subclasses only override predict(); shared utilities live here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class PredictionResult:
    """
    Output of a single predictor run over a DataLoader.

    Attributes
    ----------
    probs : np.ndarray, shape (N, C)
        Mean predictive probabilities for each sample and class.
        For the deterministic predictor this is the single softmax output.
        For MC Dropout this is the mean over T stochastic passes.
    entropy : np.ndarray, shape (N,)
        Predictive (total) entropy:  H[y | x, D] = -Σ p_c log p_c.
        High values indicate uncertain predictions.
    labels : np.ndarray, shape (N,)
        Ground-truth integer class labels from the DataLoader.
    preds : np.ndarray, shape (N,)
        Predicted class indices: argmax of probs.
    """
    probs:   np.ndarray   # (N, C)
    entropy: np.ndarray   # (N,)
    labels:  np.ndarray   # (N,)
    preds:   np.ndarray   # (N,)

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"PredictionResult("
            f"n={self.n_samples}, "
            f"mean_entropy={self.entropy.mean():.4f})"
        )


class BasePredictor(ABC):
    """
    Abstract base class for all inference modes.

    Subclasses must implement predict().  The model and config are passed
    at construction and stored on self so predict() only needs a DataLoader.

    Parameters
    ----------
    model : nn.Module
        The trained LeNet-5 instance.
    config : ExperimentConfig
        Used to read device and mc_samples.
    """

    def __init__(self, model, config) -> None:
        self.model  = model
        self.config = config
        self.device = __import__("torch").device(config.device or "cpu")

    @abstractmethod
    def predict(self, loader: DataLoader) -> PredictionResult:
        """
        Run inference over all batches in *loader*.

        Parameters
        ----------
        loader : DataLoader
            Any DataLoader whose dataset yields (image, label) pairs.

        Returns
        -------
        PredictionResult
            Aggregated predictions and uncertainty estimates for all N samples.
        """
        ...

    # ── Shared helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_entropy(probs: np.ndarray) -> np.ndarray:
        """
        Predictive entropy: H = -Σ_c p_c log(p_c).

        Parameters
        ----------
        probs : np.ndarray, shape (N, C)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        clipped = np.clip(probs, 1e-10, 1.0)
        return -np.sum(clipped * np.log(clipped), axis=1)