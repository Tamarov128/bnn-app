"""
ExperimentRunner: top-level orchestrator for a full train + evaluate run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.config import ExperimentConfig
from core.data.datasets import DataManager
from core.evaluation.evaluator import EvalResults, Evaluator
from core.inference.deterministic import DeterministicPredictor
from core.inference.mc_dropout import MCDropoutPredictor
from core.models.alexnet import AlexNetSmall
from core.registry import ModelRegistry
from core.training.trainer import Trainer, TrainingResult


@dataclass
class RunResult:
    training:             TrainingResult
    det_eval:             Optional[EvalResults] = None
    mc_eval:              Optional[EvalResults] = None
    registry_entry_name:  str                   = ""
    interrupted:          bool                  = False


class ExperimentRunner:
    """
    Coordinates the full experiment pipeline for one configuration.

    Parameters
    ----------
    config   : ExperimentConfig
    registry : ModelRegistry, optional
    on_batch_end, on_epoch_end, on_status, should_stop : callbacks
    """

    def __init__(
        self,
        config:       ExperimentConfig,
        registry:     Optional[ModelRegistry]          = None,
        on_batch_end: Optional[Callable[..., None]]    = None,
        on_epoch_end:  Optional[Callable[..., None]]   = None,
        on_status:     Optional[Callable[[str], None]] = None,
        should_stop:   Optional[Callable[[], bool]]    = None,
    ) -> None:
        self.config       = config
        self.registry     = registry or ModelRegistry()
        self.on_batch_end = on_batch_end or _noop
        self.on_epoch_end  = on_epoch_end  or _noop
        self.on_status     = on_status or (lambda msg: print(f"[Runner] {msg}"))
        self.should_stop   = should_stop or (lambda: False)
        config.ensure_dirs()

    def run(self) -> RunResult:
        self.on_status("Loading datasets…")
        data_manager = DataManager(self.config)
        train_loader = data_manager.get_train_loader()
        val_loader   = data_manager.get_val_loader()
        test_loader  = data_manager.get_test_loader()

        self.on_status("Initialising model…")
        # dropout_rate is now in TrainingConfig
        model = AlexNetSmall(dropout_rate = self.config.training.dropout_rate)

        self.on_status(
            f"Training for {self.config.training.epochs} epochs "
            f"on {self.config.device}…"
        )
        trainer = Trainer(
            config       = self.config,
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            on_batch_end = self.on_batch_end,
            on_epoch_end  = self.on_epoch_end,
            should_stop   = self.should_stop,
        )
        training_result = trainer.fit()

        if training_result.interrupted:
            self.on_status("Training interrupted.")
            return RunResult(training=training_result, interrupted=True)

        self.on_status("Saving model to registry…")
        import torch
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        entry = self.registry.save_model(
            name             = self.config.run_name,
            state_dict       = state_dict,
            config           = self.config,
            training_metrics = training_result.last_metrics(),
        )

        ood_loaders = data_manager.get_ood_loaders()

        self.on_status("Evaluating deterministic predictor…")
        det_eval = Evaluator(
            predictor      = DeterministicPredictor(model, self.config),
            predictor_name = "Deterministic",
            on_progress    = lambda s, t: self.on_status(
                f"Deterministic eval: {s}/{t} datasets"
            ),
        ).run(test_loader, ood_loaders)

        self.on_status("Evaluating MC Dropout predictor…")
        mc_eval = Evaluator(
            predictor      = MCDropoutPredictor(model, self.config),
            predictor_name = (
                f"MC Dropout (T={self.config.inference.mc_samples})"
            ),
            on_progress    = lambda s, t: self.on_status(
                f"MC Dropout eval: {s}/{t} datasets"
            ),
        ).run(test_loader, ood_loaders)

        self.on_status("Persisting evaluation metrics…")
        self.registry.update_eval_metrics(entry.name, {
            "deterministic": det_eval.to_dict(),
            "mc_dropout":    mc_eval.to_dict(),
        })

        self.on_status("Done.")
        return RunResult(
            training            = training_result,
            det_eval            = det_eval,
            mc_eval             = mc_eval,
            registry_entry_name = entry.name,
        )


def _noop(*args, **kwargs) -> None:
    pass


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BNN experiment runner")
    parser.add_argument("--run-name",   default="cli_run")
    parser.add_argument("--dataset",    default="mnist")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int,   default=50)
    parser.add_argument("--train-size", type=float, default=1.0)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    from core.config import TrainingConfig, InferenceConfig
    cfg = ExperimentConfig(
        run_name  = args.run_name,
        training  = TrainingConfig(
            train_dataset = args.dataset,
            epochs        = args.epochs,
            lr            = args.lr,
            batch_size    = args.batch_size,
            train_size    = args.train_size,
            dropout_rate  = args.dropout,
        ),
        inference = InferenceConfig(mc_samples=args.mc_samples),
        device    = args.device,
    )

    result = ExperimentRunner(config=cfg).run()
    print("\n── Training summary ───────────────────────────────")
    for k, v in result.training.last_metrics().items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    _cli()