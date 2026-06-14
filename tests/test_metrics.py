"""
tests/test_metrics.py
─────────────────────
Unit tests for core/metrics/metrics.py.

All expected values are derived analytically so the tests are
self-contained — no trained model or dataset download required.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.metrics.metrics import (
    accuracy,
    precision,
    recall,
    f1,
    ece,
    predictive_entropy,
    auroc,
    fpr_at_tpr,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures / shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def one_hot(indices: list[int], n_classes: int = 3) -> np.ndarray:
    """Return a (N, C) array of one-hot probability vectors."""
    out = np.zeros((len(indices), n_classes), dtype=np.float64)
    for i, c in enumerate(indices):
        out[i, c] = 1.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Classification metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracy:
    def test_perfect(self):
        probs  = one_hot([0, 1, 2])
        labels = np.array([0, 1, 2])
        assert accuracy(probs, labels) == pytest.approx(1.0)

    def test_zero(self):
        probs  = one_hot([1, 2, 0])      # all wrong
        labels = np.array([0, 1, 2])
        assert accuracy(probs, labels) == pytest.approx(0.0)

    def test_half(self):
        probs  = one_hot([0, 0, 1, 1])
        labels = np.array([0, 1, 0, 1])  # first and last correct
        assert accuracy(probs, labels) == pytest.approx(0.5)

    def test_return_type_is_float(self):
        probs  = one_hot([0])
        labels = np.array([0])
        assert isinstance(accuracy(probs, labels), float)


class TestPrecision:
    def test_perfect(self):
        probs  = one_hot([0, 1, 2])
        labels = np.array([0, 1, 2])
        assert precision(probs, labels) == pytest.approx(1.0)

    def test_macro_averaging(self):
        # Predict class 0 for everything; class 0 precision = 1/3,
        # classes 1 and 2 have no TPs → zero_division=0 → 0.
        # macro average = (1/3 + 0 + 0) / 3 = 1/9
        probs  = one_hot([0, 0, 0])
        labels = np.array([0, 1, 2])
        assert precision(probs, labels) == pytest.approx(1 / 9)


class TestRecall:
    def test_perfect(self):
        probs  = one_hot([0, 1, 2])
        labels = np.array([0, 1, 2])
        assert recall(probs, labels) == pytest.approx(1.0)

    def test_zero_when_all_wrong(self):
        probs  = one_hot([1, 2, 0])
        labels = np.array([0, 1, 2])
        assert recall(probs, labels) == pytest.approx(0.0)


class TestF1:
    def test_perfect(self):
        probs  = one_hot([0, 1, 2])
        labels = np.array([0, 1, 2])
        assert f1(probs, labels) == pytest.approx(1.0)

    def test_zero_when_all_wrong(self):
        probs  = one_hot([1, 2, 0])
        labels = np.array([0, 1, 2])
        assert f1(probs, labels) == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Expected Calibration Error
# ══════════════════════════════════════════════════════════════════════════════

class TestECE:
    def test_perfect_calibration(self):
        """One-hot predictions that are always correct → ECE = 0."""
        probs  = one_hot([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        assert ece(probs, labels) == pytest.approx(0.0)

    def test_uniform_probs_high_ece(self):
        """
        Uniform distribution over C=10 classes → confidence = 0.1.
        Model is wrong on all samples (labels=0, predicted argmax=0 but
        tie-break depends on implementation; here we use argmax which returns 0).
        Set labels to 1 so accuracy=0 → |0 - 0.1| = 0.1 in the bin.
        ECE should equal 0.1.
        """
        N = 100
        probs  = np.full((N, 10), 0.1, dtype=np.float64)
        labels = np.ones(N, dtype=int)          # all wrong (argmax=0)
        result = ece(probs, labels)
        assert result == pytest.approx(0.1, abs=1e-6)

    def test_output_in_unit_interval(self):
        rng    = np.random.default_rng(0)
        raw    = rng.dirichlet(np.ones(5), size=200)
        labels = rng.integers(0, 5, size=200)
        result = ece(raw, labels)
        assert 0.0 <= result <= 1.0

    def test_non_negative(self):
        probs  = one_hot([0, 1, 0, 1])
        labels = np.array([0, 0, 1, 1])
        assert ece(probs, labels) >= 0.0

    def test_analytic_two_bin(self):
        """
        Handcrafted 2-sample case with known ECE.

        Sample 0: confidence=0.9, correct   → bin gap = |1.0 - 0.9| = 0.1
        Sample 1: confidence=0.6, incorrect → bin gap = |0.0 - 0.6| = 0.6
        (they fall in different bins with n_bins=15 default)
        ECE = 0.5 * 0.1 + 0.5 * 0.6 = 0.35
        """
        probs = np.array([
            [0.9, 0.1],   # predicts class 0, correct
            [0.4, 0.6],   # predicts class 1, wrong
        ])
        labels = np.array([0, 0])
        result = ece(probs, labels)
        assert result == pytest.approx(0.35, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Predictive entropy
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictiveEntropy:
    def test_zero_entropy_for_one_hot(self):
        probs = one_hot([0, 1, 2])
        H     = predictive_entropy(probs)
        assert H == pytest.approx(np.zeros(3), abs=1e-6)

    def test_max_entropy_uniform(self):
        """Uniform over C classes → H = log(C)."""
        C     = 4
        probs = np.full((5, C), 1.0 / C)
        H     = predictive_entropy(probs)
        assert H == pytest.approx(np.full(5, np.log(C)), abs=1e-6)

    def test_output_shape(self):
        probs = np.random.default_rng(1).dirichlet(np.ones(7), size=50)
        H     = predictive_entropy(probs)
        assert H.shape == (50,)

    def test_entropy_non_negative(self):
        rng   = np.random.default_rng(2)
        probs = rng.dirichlet(np.ones(5), size=100)
        assert np.all(predictive_entropy(probs) >= 0.0)

    def test_higher_entropy_for_less_confident(self):
        """More uniform → higher entropy than peaked distribution."""
        peaked  = np.array([[0.99, 0.005, 0.005]])
        uniform = np.array([[1/3, 1/3, 1/3]])
        assert predictive_entropy(uniform)[0] > predictive_entropy(peaked)[0]


# ══════════════════════════════════════════════════════════════════════════════
# AUROC
# ══════════════════════════════════════════════════════════════════════════════

class TestAUROC:
    def test_perfect_separation(self):
        """OOD scores all higher than in-distribution → AUROC = 1.0."""
        in_scores  = np.array([0.1, 0.2, 0.3])
        ood_scores = np.array([0.8, 0.9, 1.0])
        assert auroc(in_scores, ood_scores) == pytest.approx(1.0)

    def test_random_chance(self):
        """
        Interleaved equal scores → AUROC ≈ 0.5.
        Use a large enough sample so the approximation is tight.
        """
        rng        = np.random.default_rng(42)
        in_scores  = rng.uniform(0, 1, 1000)
        ood_scores = rng.uniform(0, 1, 1000)
        result     = auroc(in_scores, ood_scores)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_worst_case(self):
        """OOD scores all lower than in-distribution → AUROC = 0.0."""
        in_scores  = np.array([0.8, 0.9, 1.0])
        ood_scores = np.array([0.1, 0.2, 0.3])
        assert auroc(in_scores, ood_scores) == pytest.approx(0.0)

    def test_output_range(self):
        rng        = np.random.default_rng(7)
        in_scores  = rng.uniform(0, 1, 50)
        ood_scores = rng.uniform(0, 1, 50)
        result     = auroc(in_scores, ood_scores)
        assert 0.0 <= result <= 1.0

    def test_return_type_is_float(self):
        result = auroc(np.array([0.2]), np.array([0.8]))
        assert isinstance(result, float)


# ══════════════════════════════════════════════════════════════════════════════
# FPR at TPR (FPR95)
# ══════════════════════════════════════════════════════════════════════════════

class TestFPRAtTPR:
    def test_perfect_separation_gives_zero_fpr(self):
        """Perfect detector: all OOD above all in-dist → FPR95 = 0."""
        in_scores  = np.linspace(0.0, 0.4, 100)
        ood_scores = np.linspace(0.6, 1.0, 100)
        result     = fpr_at_tpr(in_scores, ood_scores, tpr_threshold=0.95)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_random_gives_high_fpr(self):
        """Random scores → FPR95 should be ≈ 1 (almost all in-dist flagged)."""
        rng        = np.random.default_rng(0)
        in_scores  = rng.uniform(0, 1, 500)
        ood_scores = rng.uniform(0, 1, 500)
        result     = fpr_at_tpr(in_scores, ood_scores, tpr_threshold=0.95)
        # With random scores FPR at TPR=0.95 should be close to 0.95
        assert result == pytest.approx(0.95, abs=0.1)

    def test_fallback_when_tpr_unreachable(self):
        """If TPR threshold is never reached, return 1.0 (worst case)."""
        # All scores identical → no threshold achieves TPR ≥ 1.0
        in_scores  = np.ones(10)
        ood_scores = np.ones(10)
        result     = fpr_at_tpr(in_scores, ood_scores, tpr_threshold=1.0)
        assert result == pytest.approx(1.0)

    def test_output_non_negative(self):
        rng        = np.random.default_rng(3)
        in_scores  = rng.uniform(0, 1, 200)
        ood_scores = rng.uniform(0, 1, 200)
        assert fpr_at_tpr(in_scores, ood_scores) >= 0.0

    def test_custom_tpr_threshold(self):
        """FPR at TPR=0.5 should be tighter (lower) than FPR at TPR=0.95."""
        in_scores  = np.linspace(0.0, 0.5, 200)
        ood_scores = np.linspace(0.3, 1.0, 200)
        fpr50 = fpr_at_tpr(in_scores, ood_scores, tpr_threshold=0.50)
        fpr95 = fpr_at_tpr(in_scores, ood_scores, tpr_threshold=0.95)
        assert fpr50 <= fpr95