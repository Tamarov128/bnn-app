"""
core/inference/mc_dropout.py
─────────────────────────────
MC Dropout predictor: T stochastic forward passes with dropout active.

Implements the Gal & Ghahramani (2016) inference procedure:

    Predictive entropy  (total uncertainty):
        H[y | x, D]  =  -Σ_c p̄_c log p̄_c
        where  p̄_c = (1/T) Σ_t p_c^(t)

Memory strategy
───────────────
Accumulating the full (T, N, C) tensor in RAM for large loaders would be
expensive.  Instead we maintain a running sum over T for each batch:
    sum_probs : Σ_t p^(t)   — for mean prediction

This keeps peak memory at O(N·C) rather than O(T·N·C).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.inference.base import BasePredictor, PredictionResult


class MCDropoutPredictor(BasePredictor):
    """
    Bayesian inference via T stochastic forward passes (MC Dropout).

    Uncertainty estimates
    ─────────────────────
    entropy : predictive entropy of the mean prediction H[ȳ]

    Usage
    ─────
    >>> predictor = MCDropoutPredictor(model, config)
    >>> result    = predictor.predict(test_loader)
    >>> print(result.entropy.mean())   # mean predictive uncertainty
    """

    def predict(self, loader: DataLoader) -> PredictionResult:
        """
        Run T stochastic forward passes over all batches.

        Parameters
        ----------
        loader : DataLoader

        Returns
        -------
        PredictionResult
        """
        T = self.config.inference.mc_samples

        self.model.to(self.device)
        self.model.eval()
        self.model.enable_dropout()   # re-enable only nn.Dropout layers

        all_mean_probs: list[np.ndarray] = []
        all_labels:     list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)  # (B, 1, 28, 28)
                B      = images.size(0)

                # Running sum across T passes — shape (B, C)
                sum_probs = torch.zeros(B, 10, device=self.device)

                for _ in range(T):
                    logits     = self.model(images)       # (B, C)
                    sum_probs += F.softmax(logits, dim=1) # (B, C)

                # Mean predictive distribution: p̄ = (1/T) Σ_t p^(t)
                mean_probs = (sum_probs / T).cpu().numpy()  # (B, C)

                all_mean_probs.append(mean_probs)
                all_labels.append(labels.numpy())

        probs_arr  = np.concatenate(all_mean_probs, axis=0)   # (N, C)
        labels_arr = np.concatenate(all_labels,     axis=0)   # (N,)

        entropy = self._compute_entropy(probs_arr)             # (N,)
        preds   = np.argmax(probs_arr, axis=1)                 # (N,)

        return PredictionResult(
            probs   = probs_arr,
            entropy = entropy,
            labels  = labels_arr,
            preds   = preds,
        )