"""
core/evaluation/evaluator.py
──────────────────────────────
Evaluator: orchestrates a full evaluation run and assembles EvalResults.

Given a predictor, in-distribution test loader, and OOD loaders dict,
the Evaluator runs inference once for each dataset and computes every
metric defined in the experiment plan:

    In-distribution:  accuracy, precision, recall, F1, ECE
    Per-OOD dataset:  AUROC, FPR95  (both using predictive entropy)

EvalResults
───────────
A dataclass that is directly serialisable to JSON (via to_dict()) and
consumed by the GUI's metrics table and plot widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.inference.base import BasePredictor, PredictionResult
from core.metrics.metrics import accuracy, precision, recall, f1, ece, auroc, fpr_at_tpr


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class InDistResults:
    """In-distribution classification and calibration metrics."""
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    ece:       float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class OODResults:
    """OOD detection metrics for a single OOD dataset."""
    dataset_name:     str
    auroc:            float
    fpr95:            float
    mean_entropy_ood: float = 0.0
    mean_entropy_in:  float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class EvalResults:
    """
    Complete evaluation output for one predictor on one experiment run.

    Attributes
    ----------
    predictor_name : str
        Human-readable label, e.g. "Deterministic" or "MC Dropout (T=50)".
    in_dist : InDistResults
    ood : dict[str, OODResults]
        Keyed by OOD dataset name, e.g. "fashion_mnist".
    in_dist_prediction : PredictionResult
        Raw predictions on the test set — kept for plot widgets that need
        the full entropy / probability arrays (e.g. histograms).
    ood_predictions : dict[str, PredictionResult]
        Raw predictions for each OOD set.
    """
    predictor_name:      str
    in_dist:             InDistResults
    ood:                 dict[str, OODResults]      = field(default_factory=dict)
    in_dist_prediction:  Optional[PredictionResult] = field(default=None, repr=False)
    ood_predictions:     dict[str, PredictionResult] = field(
        default_factory=dict, repr=False
    )

    def to_dict(self) -> dict:
        """JSON-serialisable summary (omits raw PredictionResult arrays)."""
        return {
            "predictor_name": self.predictor_name,
            "in_dist":        self.in_dist.to_dict(),
            "ood":            {k: v.to_dict() for k, v in self.ood.items()},
        }

    def ood_table_rows(self) -> list[dict]:
        """
        Flat list of dicts suitable for populating a QTableWidget.
        One row per OOD dataset.
        """
        rows = []
        for name, r in self.ood.items():
            rows.append({
                "Dataset": name.replace("_", " ").title(),
                "AUROC":   f"{r.auroc:.4f}",
                "FPR95":   f"{r.fpr95:.4f}",
            })
        return rows


# ── Evaluator ──────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs inference and computes all evaluation metrics.

    Parameters
    ----------
    predictor : BasePredictor
        Any concrete predictor (deterministic or MC Dropout).
    predictor_name : str
        Label stored in EvalResults.
    on_progress : callable, optional
        Called as on_progress(step, total) after each dataset is evaluated.
        Used by InferenceWorker to drive a progress bar.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        predictor_name: str = "predictor",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.predictor      = predictor
        self.predictor_name = predictor_name
        self.on_progress    = on_progress or (lambda s, t: None)

    def run(
        self,
        test_loader,
        ood_loaders: dict,
    ) -> EvalResults:
        """
        Execute the full evaluation pipeline.

        Parameters
        ----------
        test_loader : DataLoader
            In-distribution MNIST test set.
        ood_loaders : dict[str, DataLoader]
            OOD DataLoaders keyed by dataset name.

        Returns
        -------
        EvalResults
        """
        total_steps = 1 + len(ood_loaders)
        step = 0

        # ── In-distribution ───────────────────────────────────────────────────
        in_result = self.predictor.predict(test_loader)
        step += 1
        self.on_progress(step, total_steps)

        in_dist = InDistResults(
            accuracy  = accuracy( in_result.probs, in_result.labels),
            precision = precision(in_result.probs, in_result.labels),
            recall    = recall(   in_result.probs, in_result.labels),
            f1        = f1(       in_result.probs, in_result.labels),
            ece       = ece(      in_result.probs, in_result.labels),
        )

        # ── OOD ───────────────────────────────────────────────────────────────
        ood_results:     dict[str, OODResults]       = {}
        ood_predictions: dict[str, PredictionResult] = {}

        in_entropy_scores = in_result.entropy   # higher entropy → OOD

        for name, loader in ood_loaders.items():
            ood_result = self.predictor.predict(loader)

            ood_results[name] = OODResults(
                dataset_name     = name,
                auroc            = auroc(   in_entropy_scores, ood_result.entropy),
                fpr95            = fpr_at_tpr(in_entropy_scores, ood_result.entropy),
                mean_entropy_ood = float(ood_result.entropy.mean()),
                mean_entropy_in  = float(in_result.entropy.mean()),
            )
            ood_predictions[name] = ood_result

            step += 1
            self.on_progress(step, total_steps)

        return EvalResults(
            predictor_name     = self.predictor_name,
            in_dist            = in_dist,
            ood                = ood_results,
            in_dist_prediction = in_result,
            ood_predictions    = ood_predictions,
        )