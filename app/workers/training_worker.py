"""
app/workers/training_worker.py
───────────────────────────────
QThread subclass that runs the entire training loop off the main thread.

Delegates all loop logic to core.training.trainer.Trainer, passing Qt
signal emitters as callbacks.  This keeps training logic in the pure-Python
backend and the worker thin.

Signal contract
───────────────
  epoch_completed(epoch: int, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float)
      Emitted at the end of every epoch.  Connect to the live plot widget
      to update loss / accuracy curves in real time.

  batch_completed(epoch: int, batch: int, total_batches: int, loss: float)
      Emitted after each mini-batch.  Drives a fine-grained progress bar.

  training_finished(state_dict: dict, final_metrics: dict)
      Emitted once on successful completion.  Carries the model state_dict
      (weights moved to CPU) and a flat summary of the final epoch metrics
      so the GUI can prompt the user to save without extra computation.

  error_occurred(message: str)
      Emitted on any unhandled exception.  Carries the full formatted
      traceback so the GUI can display an actionable QMessageBox.

  progress_updated(value: int, maximum: int)
      Generic integer progress for a QProgressBar.
      value   = current step (epoch x batches_per_epoch + batch_idx)
      maximum = total steps  (epochs x batches_per_epoch)

  status_updated(message: str)
      Free-text status messages for the GUI status bar
      ("Epoch 3/20 — val_acc: 0.9821", etc.).

Usage
─────
    worker = TrainingWorker(config=cfg, model=model,
                            train_loader=tl, val_loader=vl)
    worker.epoch_completed.connect(plot_widget.update_curves)
    worker.training_finished.connect(on_training_done)
    worker.error_occurred.connect(on_error)
    worker.start()          # non-blocking

    # later, if the user presses Stop:
    worker.request_stop()
"""

from __future__ import annotations

import traceback

import torch.nn as nn
from torch.utils.data import DataLoader
from PyQt6.QtCore import QThread, pyqtSignal

from core.config import ExperimentConfig
from core.training.trainer import Trainer


class TrainingWorker(QThread):
    """
    Thin Qt wrapper around core.training.trainer.Trainer.

    Parameters
    ----------
    config       : ExperimentConfig
    model        : nn.Module
        The model to train in-place.  The caller retains ownership;
        the final state_dict is returned via training_finished.
    train_loader : DataLoader
    val_loader   : DataLoader
    """

    # ── Signals ────────────────────────────────────────────────────────────────

    epoch_completed = pyqtSignal(int, float, float, float, float)
    # (epoch, train_loss, val_loss, train_acc, val_acc)

    batch_completed = pyqtSignal(int, int, int, float)
    # (epoch, batch_idx, total_batches, batch_loss)

    training_finished = pyqtSignal(dict, dict)
    # (state_dict, final_metrics)

    error_occurred = pyqtSignal(str)
    # (traceback_string)

    progress_updated = pyqtSignal(int, int)
    # (current_step, total_steps)

    status_updated = pyqtSignal(str)
    # (message)

    # ── Init ───────────────────────────────────────────────────────────────────

    def __init__(
        self,
        config:       ExperimentConfig,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config       = config
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self._stop_flag   = False

        # Pre-compute total steps so the progress bar maximum is known
        # before training starts.
        self._total_steps  = config.training.epochs * len(train_loader)
        self._current_step = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        """
        Ask the worker to stop after the current batch completes.

        training_finished is NOT emitted after a stop; the GUI should
        treat an interrupted run as unsaved and update its controls
        accordingly.
        """
        self._stop_flag = True

    # ── Thread entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Called by QThread.start().  Do not call directly."""
        try:
            self._run_training()
        except Exception:
            self.error_occurred.emit(traceback.format_exc())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_training(self) -> None:
        trainer = Trainer(
            config       = self.config,
            model        = self.model,
            train_loader = self.train_loader,
            val_loader   = self.val_loader,
            on_batch_end = self._on_batch,
            on_epoch_end  = self._on_epoch,
            should_stop   = lambda: self._stop_flag,
        )

        self.status_updated.emit("Training started…")
        result = trainer.fit()

        if result.interrupted:
            self.status_updated.emit("Training stopped by user.")
            return

        # Move all tensors to CPU before crossing the thread boundary.
        state_dict = {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }
        self.status_updated.emit("Training complete.")
        self.training_finished.emit(state_dict, result.last_metrics())

    def _on_batch(
        self,
        epoch:         int,
        batch_idx:     int,
        total_batches: int,
        loss:          float,
    ) -> None:
        """Forwarded from Trainer.on_batch_end."""
        self._current_step += 1
        self.batch_completed.emit(epoch, batch_idx, total_batches, loss)
        self.progress_updated.emit(self._current_step, self._total_steps)

    def _on_epoch(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
        train_acc:  float,
        val_acc:    float,
    ) -> None:
        """Forwarded from Trainer.on_epoch_end."""
        self.epoch_completed.emit(epoch, train_loss, val_loss, train_acc, val_acc)
        self.status_updated.emit(
            f"Epoch {epoch + 1}/{self.config.training.epochs} — "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"val_acc: {val_acc:.4f}"
        )