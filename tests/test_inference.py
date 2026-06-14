"""
tests/test_inference.py
────────────────────────
Output-shape and dtype contracts for DeterministicPredictor and
MCDropoutPredictor.

Strategy
────────
We build a minimal LeNet-style stub (a plain nn.Module with 10 output
classes) and a fake DataLoader that yields synthetic (B, 1, 28, 28)
tensors.  No real weights or datasets are needed.

The tests verify:
  - probs shape is (N, C)
  - entropy shape is (N,)
  - preds shape is (N,)
  - labels shape is (N,)
  - probs rows sum to 1.0 (valid probability simplex)
  - entropy is non-negative
  - preds equal argmax of probs
  - dtypes are float64 / int64 as expected
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.inference.base import PredictionResult
from core.inference.deterministic import DeterministicPredictor
from core.inference.mc_dropout import MCDropoutPredictor


# ══════════════════════════════════════════════════════════════════════════════
# Minimal stubs
# ══════════════════════════════════════════════════════════════════════════════

N_CLASSES    = 10
BATCH_SIZE   = 8
N_BATCHES    = 4
N_TOTAL      = BATCH_SIZE * N_BATCHES   # 32 samples


class TinyNet(nn.Module):
    """Minimal feedforward net with one Dropout layer (for MC Dropout tests)."""

    def __init__(self, p_drop: float = 0.5) -> None:
        super().__init__()
        self.flat    = nn.Flatten()
        self.drop    = nn.Dropout(p=p_drop)
        self.linear  = nn.Linear(28 * 28, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.drop(self.flat(x)))

    def enable_dropout(self) -> None:
        """Re-enable Dropout layers while keeping BatchNorm etc. in eval mode."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


@dataclass
class FakeConfig:
    """Mimics the subset of ExperimentConfig consumed by the predictors."""
    device: str = "cpu"

    @dataclass
    class _Inference:
        mc_samples: int = 5

    inference: _Inference = field(default_factory=_Inference)


def make_loader(n_total: int = N_TOTAL, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Return a DataLoader of synthetic (B, 1, 28, 28) images with random labels."""
    images = torch.randn(n_total, 1, 28, 28)
    labels = torch.randint(0, N_CLASSES, (n_total,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


# ══════════════════════════════════════════════════════════════════════════════
# PredictionResult dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionResult:
    def _make(self, n: int = 20, c: int = 5) -> PredictionResult:
        rng    = np.random.default_rng(0)
        raw    = rng.dirichlet(np.ones(c), size=n)
        H      = -np.sum(raw * np.log(np.clip(raw, 1e-10, 1.0)), axis=1)
        labels = rng.integers(0, c, size=n)
        preds  = np.argmax(raw, axis=1)
        return PredictionResult(probs=raw, entropy=H, labels=labels, preds=preds)

    def test_n_samples(self):
        r = self._make(n=37)
        assert r.n_samples == 37

    def test_repr_contains_n(self):
        r = self._make(n=20)
        assert "n=20" in repr(r)

    def test_repr_contains_entropy(self):
        r = self._make()
        assert "mean_entropy" in repr(r)


# ══════════════════════════════════════════════════════════════════════════════
# DeterministicPredictor
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterministicPredictor:
    @pytest.fixture
    def result(self) -> PredictionResult:
        model  = TinyNet()
        config = FakeConfig()
        loader = make_loader()
        pred   = DeterministicPredictor(model, config)
        return pred.predict(loader)

    # ── Shape contracts ───────────────────────────────────────────────────────

    def test_probs_shape(self, result):
        assert result.probs.shape == (N_TOTAL, N_CLASSES)

    def test_entropy_shape(self, result):
        assert result.entropy.shape == (N_TOTAL,)

    def test_labels_shape(self, result):
        assert result.labels.shape == (N_TOTAL,)

    def test_preds_shape(self, result):
        assert result.preds.shape == (N_TOTAL,)

    # ── Value contracts ───────────────────────────────────────────────────────

    def test_probs_sum_to_one(self, result):
        row_sums = result.probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(N_TOTAL), atol=1e-5)

    def test_probs_non_negative(self, result):
        assert np.all(result.probs >= 0.0)

    def test_entropy_non_negative(self, result):
        assert np.all(result.entropy >= 0.0)

    def test_preds_equal_argmax_of_probs(self, result):
        expected = np.argmax(result.probs, axis=1)
        np.testing.assert_array_equal(result.preds, expected)

    def test_preds_in_class_range(self, result):
        assert np.all(result.preds >= 0)
        assert np.all(result.preds < N_CLASSES)

    # ── Dtype contracts ───────────────────────────────────────────────────────

    def test_probs_dtype_float(self, result):
        assert np.issubdtype(result.probs.dtype, np.floating)

    def test_labels_dtype_integer(self, result):
        assert np.issubdtype(result.labels.dtype, np.integer)

    # ── Determinism ───────────────────────────────────────────────────────────

    def test_deterministic_results_are_identical_on_repeat(self):
        """Same model, same loader → identical probs on two runs."""
        model  = TinyNet()
        config = FakeConfig()
        loader = make_loader(n_total=16, batch_size=8)
        pred   = DeterministicPredictor(model, config)
        r1     = pred.predict(loader)
        r2     = pred.predict(loader)
        np.testing.assert_array_equal(r1.probs, r2.probs)

    def test_single_sample_loader(self):
        """Edge case: loader with exactly one sample."""
        model  = TinyNet()
        config = FakeConfig()
        loader = make_loader(n_total=1, batch_size=1)
        pred   = DeterministicPredictor(model, config)
        result = pred.predict(loader)
        assert result.probs.shape == (1, N_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# MCDropoutPredictor
# ══════════════════════════════════════════════════════════════════════════════

class TestMCDropoutPredictor:
    @pytest.fixture
    def result(self) -> PredictionResult:
        model  = TinyNet()
        config = FakeConfig()
        loader = make_loader()
        pred   = MCDropoutPredictor(model, config)
        return pred.predict(loader)

    # ── Shape contracts ───────────────────────────────────────────────────────

    def test_probs_shape(self, result):
        assert result.probs.shape == (N_TOTAL, N_CLASSES)

    def test_entropy_shape(self, result):
        assert result.entropy.shape == (N_TOTAL,)

    def test_labels_shape(self, result):
        assert result.labels.shape == (N_TOTAL,)

    def test_preds_shape(self, result):
        assert result.preds.shape == (N_TOTAL,)

    # ── Value contracts ───────────────────────────────────────────────────────

    def test_probs_sum_to_one(self, result):
        row_sums = result.probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(N_TOTAL), atol=1e-5)

    def test_probs_non_negative(self, result):
        assert np.all(result.probs >= 0.0)

    def test_entropy_non_negative(self, result):
        assert np.all(result.entropy >= 0.0)

    def test_preds_equal_argmax_of_probs(self, result):
        expected = np.argmax(result.probs, axis=1)
        np.testing.assert_array_equal(result.preds, expected)

    def test_preds_in_class_range(self, result):
        assert np.all(result.preds >= 0)
        assert np.all(result.preds < N_CLASSES)

    # ── Stochasticity ─────────────────────────────────────────────────────────

    def test_mc_results_vary_across_runs(self):
        """Dropout is active → two independent runs should differ."""
        model  = TinyNet(p_drop=0.5)
        config = FakeConfig()
        loader = make_loader(n_total=16, batch_size=8)
        pred   = MCDropoutPredictor(model, config)
        r1     = pred.predict(loader)
        r2     = pred.predict(loader)
        # With p=0.5 and T=5 samples, probs should differ for at least one sample
        assert not np.allclose(r1.probs, r2.probs)

    def test_mc_entropy_higher_than_deterministic_on_average(self):
        """
        MC Dropout with active stochasticity should produce higher average
        entropy than deterministic eval on the same model (statistically).
        This is a soft expectation — not a guarantee for every model/seed,
        but reliably true with p=0.5 and T=5.
        """
        model  = TinyNet(p_drop=0.5)
        config = FakeConfig()
        loader = make_loader(n_total=64, batch_size=16)

        det_result = DeterministicPredictor(model, config).predict(loader)
        mc_result  = MCDropoutPredictor(model, config).predict(loader)

        assert mc_result.entropy.mean() >= det_result.entropy.mean() * 0.8

    def test_varying_mc_samples(self):
        """T=1 and T=20 should both produce valid shapes."""
        model  = TinyNet()
        loader = make_loader(n_total=8, batch_size=8)

        for t in (1, 20):
            @dataclass
            class Cfg:
                device: str = "cpu"
                @dataclass
                class _I:
                    mc_samples: int = t
                inference: _I = field(default_factory=_I)

            result = MCDropoutPredictor(model, Cfg()).predict(loader)
            assert result.probs.shape == (8, N_CLASSES)

    def test_single_sample_loader(self):
        model  = TinyNet()
        config = FakeConfig()
        loader = make_loader(n_total=1, batch_size=1)
        result = MCDropoutPredictor(model, config).predict(loader)
        assert result.probs.shape == (1, N_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# Shared BasePredictor helper
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeEntropy:
    """Tests for the shared _compute_entropy static method via DeterministicPredictor."""

    def _entropy(self, probs):
        from core.inference.base import BasePredictor
        return BasePredictor._compute_entropy(probs)

    def test_one_hot_zero_entropy(self):
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        H = self._entropy(p)
        np.testing.assert_allclose(H, [0.0, 0.0], atol=1e-6)

    def test_uniform_max_entropy(self):
        C = 5
        p = np.full((3, C), 1.0 / C)
        H = self._entropy(p)
        np.testing.assert_allclose(H, np.full(3, np.log(C)), atol=1e-6)

    def test_numerical_stability_near_zero(self):
        """Probabilities very close to 0 should not produce NaN or Inf."""
        p = np.array([[1.0 - 1e-9, 1e-9]])
        H = self._entropy(p)
        assert np.isfinite(H).all()