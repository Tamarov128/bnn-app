"""
core/training/trainer.py
─────────────────────────
Pure-Python training loop with no Qt dependency.

The Trainer class owns the epoch/batch loop and reports progress through
optional callbacks rather than Qt signals.  This keeps it independently
testable from the CLI and decouples training logic from the GUI layer.

The TrainingWorker (app/workers/training_worker.py) wraps this class and
passes its pyqtSignal.emit methods as callbacks, so the same logic drives
both GUI and headless execution.

Callback signatures
───────────────────
  on_batch_end(epoch: int, batch: int, total_batches: int, loss: float) -> None
  on_epoch_end(epoch: int, train_loss: float, val_loss: float,
               train_acc: float,  val_acc: float) -> None
  should_stop() -> bool
      Polled once per batch.  Return True to abort training cleanly.

TrainingResult
──────────────
Dataclass returned by Trainer.fit().  Contains the per-epoch history
and best validation accuracy observed during training.

Input note
──────────
AlexNetSmall expects 3-channel inputs (B, 3, 28, 28).  If your DataLoader
yields greyscale MNIST tensors (B, 1, 28, 28), replicate the channel
dimension in your dataset transform or at the top of _train_epoch /
_val_epoch:
    images = images.repeat(1, 3, 1, 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.config import ExperimentConfig


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """
    Outcome of a completed (or interrupted) training run.

    Attributes
    ----------
    train_losses, val_losses : list[float]
        Per-epoch average losses.
    train_accs, val_accs : list[float]
        Per-epoch accuracy (fraction in [0, 1]).
    best_val_acc : float
        Highest validation accuracy seen across all epochs.
    best_epoch : int
        0-indexed epoch at which best_val_acc was achieved.
    epochs_completed : int
        Number of fully completed epochs (may be < config.epochs if stopped).
    interrupted : bool
        True if training was halted early via should_stop().
    """
    train_losses:     list[float] = field(default_factory=list)
    val_losses:       list[float] = field(default_factory=list)
    train_accs:       list[float] = field(default_factory=list)
    val_accs:         list[float] = field(default_factory=list)
    best_val_acc:     float       = 0.0
    best_epoch:       int         = 0
    epochs_completed: int         = 0
    interrupted:      bool        = False

    def last_metrics(self) -> dict:
        """Return a flat dict of the final epoch's metrics."""
        if not self.train_losses:
            return {}
        return {
            "epoch":          self.epochs_completed - 1,
            "train_loss":     self.train_losses[-1],
            "val_loss":       self.val_losses[-1],
            "train_acc":      self.train_accs[-1],
            "val_acc":        self.val_accs[-1],
            "best_val_acc":   self.best_val_acc,
            "best_epoch":     self.best_epoch,
        }


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the training and validation loops for LeNet-5.

    Parameters
    ----------
    config : ExperimentConfig
        Source of hyperparameters (epochs, lr, weight_decay, device).
    model : nn.Module
        The model to train.  Moved to config.device at the start of fit().
    train_loader : DataLoader
    val_loader   : DataLoader
    on_batch_end : callable, optional
        Called after every mini-batch with
        (epoch, batch_idx, total_batches, batch_loss).
    on_epoch_end : callable, optional
        Called after every epoch with
        (epoch, train_loss, val_loss, train_acc, val_acc).
    should_stop : callable, optional
        Polled after every batch; return True to interrupt training.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        on_batch_end: Optional[Callable[..., None]] = None,
        on_epoch_end:  Optional[Callable[..., None]] = None,
        should_stop:   Optional[Callable[[], bool]]  = None,
    ) -> None:
        self.config       = config
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.on_batch_end = on_batch_end or _noop
        self.on_epoch_end  = on_epoch_end  or _noop
        self.should_stop   = should_stop   or (lambda: False)

        self.device = torch.device(config.device or "cpu")

    # ── Main entry point ───────────────────────────────────────────────────────

    def fit(self) -> TrainingResult:
        """
        Run the full training loop.

        Returns
        -------
        TrainingResult
            History and summary of the completed run.
        """
        self.model.to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        # Cosine annealing with linear warm-up: smooth LR schedule that
        # works well for MNIST without requiring manual tuning.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.lr,
            epochs=self.config.training.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,       # 10 % of steps used for warm-up
            anneal_strategy="cos",
        )

        criterion = nn.CrossEntropyLoss()
        result    = TrainingResult()
        n_epochs  = self.config.training.epochs

        for epoch in range(n_epochs):
            # ── Train ─────────────────────────────────────────────────────────
            t_loss, t_acc, interrupted = self._train_epoch(
                epoch, optimizer, scheduler, criterion
            )

            if interrupted:
                result.interrupted = True
                break

            # ── Validate ──────────────────────────────────────────────────────
            v_loss, v_acc = self._val_epoch(criterion)

            # ── Record ────────────────────────────────────────────────────────
            result.train_losses.append(t_loss)
            result.val_losses.append(v_loss)
            result.train_accs.append(t_acc)
            result.val_accs.append(v_acc)
            result.epochs_completed += 1

            if v_acc > result.best_val_acc:
                result.best_val_acc = v_acc
                result.best_epoch   = epoch

            self.on_epoch_end(epoch, t_loss, v_loss, t_acc, v_acc)

        return result

    # ── Epoch helpers ──────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler,
        criterion: nn.Module,
    ) -> tuple[float, float, bool]:
        """
        One training epoch.

        Returns
        -------
        avg_loss : float
        accuracy : float
        interrupted : bool
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        n_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if self.should_stop():
                return total_loss / max(total, 1), correct / max(total, 1), True

            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping — prevents occasional spikes early in training.
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            batch_loss   = loss.item()
            total_loss  += batch_loss * images.size(0)
            preds        = logits.argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += images.size(0)

            self.on_batch_end(epoch, batch_idx, n_batches, batch_loss)

        return total_loss / max(total, 1), correct / max(total, 1), False

    def _val_epoch(
        self,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """One validation pass.  Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                
                images, labels = images.to(self.device), labels.to(self.device)
                logits     = self.model(images)
                loss       = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _noop(*args, **kwargs) -> None:
    """Default no-op callback."""
    pass