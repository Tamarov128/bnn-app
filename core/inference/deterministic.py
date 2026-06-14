"""
core/inference/deterministic.py
─────────────────────────────────
Deterministic single-pass predictor (standard neural network baseline).

Runs one forward pass per batch with dropout fully disabled (model.eval()).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.inference.base import BasePredictor, PredictionResult


class DeterministicPredictor(BasePredictor):
    """
    Single deterministic forward pass predictor.

    Uncertainty estimates
    ─────────────────────
    entropy : H[y | x] = -Σ p_c log p_c

    Usage
    ─────
    >>> predictor = DeterministicPredictor(model, config)
    >>> result    = predictor.predict(test_loader)
    """

    def predict(self, loader: DataLoader) -> PredictionResult:
        """
        Run a single forward pass over all batches.

        Parameters
        ----------
        loader : DataLoader
            Yields (image_tensor, label_tensor) batches.

        Returns
        -------
        PredictionResult
        """
        self.model.to(self.device)
        self.model.eval()

        all_probs:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)

                logits = self.model(images)                        # (B, C)
                probs  = F.softmax(logits, dim=1).cpu().numpy()   # (B, C)

                all_probs.append(probs)
                all_labels.append(labels.numpy())

        probs_arr  = np.concatenate(all_probs,  axis=0)   # (N, C)
        labels_arr = np.concatenate(all_labels, axis=0)   # (N,)

        entropy = self._compute_entropy(probs_arr)         # (N,)
        preds   = np.argmax(probs_arr, axis=1)             # (N,)

        return PredictionResult(
            probs   = probs_arr,
            entropy = entropy,
            labels  = labels_arr,
            preds   = preds,
        )