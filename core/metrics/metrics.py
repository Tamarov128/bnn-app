"""
core/metrics/metrics.py
───────────────────────
Classification and OOD detection metrics.

Classification metrics:
    accuracy, precision, recall, f1  — standard prediction quality scores.
    ece                              — Expected Calibration Error.

OOD detection metrics:
    predictive_entropy  — Shannon entropy used as an uncertainty score.
    auroc               — Area Under the ROC Curve.
    fpr_at_tpr          — FPR at a given TPR level (default: FPR95).

All functions accept plain NumPy arrays — no PyTorch, no Qt.
OOD convention: higher score = more likely OOD; OOD is the positive class.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ──────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────

def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Fraction of correctly classified samples.

    Parameters
    ----------
    probs  : np.ndarray, shape (N, C)
    labels : np.ndarray, shape (N,)  integer class indices

    Returns
    -------
    float in [0, 1].  Higher is better.
    """
    predictions = np.argmax(probs, axis=1)
    return float(accuracy_score(labels, predictions))


def precision(
    probs: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Precision score.

    Parameters
    ----------
    probs   : np.ndarray, shape (N, C)
    labels  : np.ndarray, shape (N,)
    average : str  sklearn averaging strategy (default 'macro').

    Returns
    -------
    float in [0, 1].  Higher is better.
    """
    predictions = np.argmax(probs, axis=1)
    return float(precision_score(labels, predictions, average=average, zero_division=0))


def recall(
    probs: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Recall score.

    Parameters
    ----------
    probs   : np.ndarray, shape (N, C)
    labels  : np.ndarray, shape (N,)
    average : str  sklearn averaging strategy (default 'macro').

    Returns
    -------
    float in [0, 1].  Higher is better.
    """
    predictions = np.argmax(probs, axis=1)
    return float(recall_score(labels, predictions, average=average, zero_division=0))


def f1(
    probs: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> float:
    """
    F1 score.

    Parameters
    ----------
    probs   : np.ndarray, shape (N, C)
    labels  : np.ndarray, shape (N,)
    average : str  sklearn averaging strategy (default 'macro').

    Returns
    -------
    float in [0, 1].  Higher is better.
    """
    predictions = np.argmax(probs, axis=1)
    return float(f1_score(labels, predictions, average=average, zero_division=0))


def ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error.

    Measures the weighted average gap between confidence and accuracy
    across M equal-width bins.  Lower is better; 0 is perfect calibration.

    Parameters
    ----------
    probs  : np.ndarray, shape (N, C)
        Predicted class probability distributions.
    labels : np.ndarray, shape (N,)
        Ground-truth integer class indices.
    n_bins : int
        Number of equal-width confidence bins in [0, 1].

    Returns
    -------
    float in [0, 1].  Lower is better.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val   = 0.0
    N         = len(labels)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += (n_bin / N) * abs(bin_acc - bin_conf)

    return float(ece_val)


# ──────────────────────────────────────────────
# OOD detection metrics
# ──────────────────────────────────────────────

def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of the predictive distribution.

    Used as an uncertainty score for OOD detection: higher entropy
    indicates higher uncertainty, i.e. more likely OOD.

    Parameters
    ----------
    probs : np.ndarray, shape (N, C)
        Class probability vectors (must sum to 1 along axis=1).

    Returns
    -------
    np.ndarray, shape (N,)
        H[y] = -Σ_c p_c log p_c  per sample.
    """
    clipped = np.clip(probs, 1e-10, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def auroc(
    in_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """
    Area Under the ROC Curve for OOD detection.

    Concatenates in-distribution scores (label=0) and OOD scores (label=1)
    and computes AUROC.  1.0 means perfect separation; 0.5 is random chance.

    Parameters
    ----------
    in_scores  : np.ndarray, shape (N_in,)
        Uncertainty scores for in-distribution samples.
    ood_scores : np.ndarray, shape (N_ood,)
        Uncertainty scores for OOD samples.

    Returns
    -------
    float in [0, 1].  Higher is better.
    """
    scores = np.concatenate([in_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(in_scores),  dtype=int),
        np.ones( len(ood_scores), dtype=int),
    ])
    return float(roc_auc_score(labels, scores))


def fpr_at_tpr(
    in_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr_threshold: float = 0.95,
) -> float:
    """
    False positive rate at a given true positive rate (FPR95 by default).

    Finds the decision threshold at which TPR ≥ tpr_threshold, then returns
    the corresponding FPR.

    Parameters
    ----------
    in_scores      : np.ndarray, shape (N_in,)
    ood_scores     : np.ndarray, shape (N_ood,)
    tpr_threshold  : float  (default 0.95)

    Returns
    -------
    float.  Lower is better.
    """
    scores = np.concatenate([in_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(in_scores),  dtype=int),
        np.ones( len(ood_scores), dtype=int),
    ])
    fprs, tprs, _ = roc_curve(labels, scores)

    indices = np.where(tprs >= tpr_threshold)[0]
    if len(indices) == 0:
        return 1.0  # threshold never reached
    return float(fprs[indices[0]])