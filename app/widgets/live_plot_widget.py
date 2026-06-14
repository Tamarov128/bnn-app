"""
app/widgets/live_plot_widget.py
────────────────────────────────
LivePlotWidget: embedded matplotlib figure for real-time training curves.

Displays loss and accuracy on two vertically stacked axes.  The widget
exposes a single public method — update(epoch, train_loss, val_loss,
train_acc, val_acc) — that is connected directly to the TrainingWorker's
epoch_completed signal.

Design notes
────────────
- Uses the dark palette from the QSS theme so the figure integrates
  seamlessly with the application window.
- Axes are drawn without frames; only left and bottom spines are kept,
  styled as subtle grid lines.
- A legend in the top-right corner distinguishes train vs. val curves.
- The figure is redrawn via canvas.draw_idle() (not draw()) to avoid
  blocking the Qt event loop.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("QtAgg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

# ── Palette (mirrors style.qss) ────────────────────────────────────────────────
_BG       = "#0d0f12"
_SURFACE  = "#161920"
_GRID     = "#2a2e38"
_TEXT_DIM = "#555d6e"
_TEXT_SEC = "#8b92a5"
_ACCENT   = "#f5a623"
_SUCCESS  = "#4caf7d"
_ERROR    = "#e05c5c"
_WHITE    = "#e8eaed"

# Train / val colour pairs  (loss, accuracy)
_TRAIN_LOSS_COLOR = _ACCENT          # amber
_VAL_LOSS_COLOR   = "#e8a838"        # warm yellow
_TRAIN_ACC_COLOR  = _SUCCESS         # green
_VAL_ACC_COLOR    = "#87ceab"        # lighter green


class LivePlotWidget(QWidget):
    """
    Dual-axis live training plot embedded in a Qt widget.

    Axes
    ────
    Top    : Cross-entropy loss   (train = amber, val = yellow)
    Bottom : Accuracy             (train = green, val = light green)

    Usage
    ─────
    >>> plot = LivePlotWidget()
    >>> worker.epoch_completed.connect(plot.on_epoch)
    >>> plot.reset()   # call before starting a new training run
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._epochs:     list[int]   = []
        self._train_loss: list[float] = []
        self._val_loss:   list[float] = []
        self._train_acc:  list[float] = []
        self._val_acc:    list[float] = []

        self._build_figure()
        self._build_layout()

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_epoch(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
        train_acc:  float,
        val_acc:    float,
    ) -> None:
        """
        Append one epoch of data and redraw.
        Connect to TrainingWorker.epoch_completed.
        """
        self._epochs.append(epoch + 1)           # display 1-indexed
        self._train_loss.append(train_loss)
        self._val_loss.append(val_loss)
        self._train_acc.append(train_acc)
        self._val_acc.append(val_acc)
        self._redraw()

    def reset(self) -> None:
        """Clear all data and redraw empty axes.  Call before each new run."""
        self._epochs.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        self._train_acc.clear()
        self._val_acc.clear()
        self._redraw()

    # ── Figure construction ────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self._fig = Figure(figsize=(6, 4), facecolor=_BG)

        self._fig.subplots_adjust(
            bottom=0.15,
            wspace=0.08,
        )

        # Two subplots side by side
        self._ax_loss, self._ax_acc = self._fig.subplots(
            1, 2,
            gridspec_kw={"wspace": 0.25}
        )

        for ax in (self._ax_loss, self._ax_acc):
            ax.set_facecolor(_SURFACE)
            ax.tick_params(colors=_TEXT_DIM, labelsize=9)
            ax.yaxis.label.set_color(_TEXT_SEC)
            ax.xaxis.label.set_color(_TEXT_SEC)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_color(_GRID)
            ax.spines["bottom"].set_color(_GRID)
            ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--", alpha=0.6)

        self._ax_loss.set_ylabel("Loss", color=_TEXT_SEC, fontsize=10)
        self._ax_acc.set_ylabel("Accuracy", color=_TEXT_SEC, fontsize=10)
        self._ax_loss.set_xlabel("Epoch", color=_TEXT_SEC, fontsize=10)
        self._ax_acc.set_xlabel("Epoch", color=_TEXT_SEC, fontsize=10)
        self._ax_acc.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
        )

        # Initialise empty line handles so we can call set_data() later.
        (self._line_tl,) = self._ax_loss.plot(
            [], [], color=_TRAIN_LOSS_COLOR, linewidth=1.8,
            label="train", solid_capstyle="round"
        )
        (self._line_vl,) = self._ax_loss.plot(
            [], [], color=_VAL_LOSS_COLOR, linewidth=1.8,
            label="val", linestyle="--", solid_capstyle="round"
        )
        (self._line_ta,) = self._ax_acc.plot(
            [], [], color=_TRAIN_ACC_COLOR, linewidth=1.8,
            label="train", solid_capstyle="round"
        )
        (self._line_va,) = self._ax_acc.plot(
            [], [], color=_VAL_ACC_COLOR, linewidth=1.8,
            label="val", linestyle="--", solid_capstyle="round"
        )

        _legend_kw = dict(
            facecolor=_SURFACE, edgecolor=_GRID,
            labelcolor=_TEXT_SEC, fontsize=9,
            loc="upper right",
        )
        self._ax_loss.legend(**_legend_kw)
        self._ax_acc.legend(**_legend_kw)

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    # ── Redraw ─────────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        ep = self._epochs

        self._line_tl.set_data(ep, self._train_loss)
        self._line_vl.set_data(ep, self._val_loss)
        self._line_ta.set_data(ep, self._train_acc)
        self._line_va.set_data(ep, self._val_acc)

        for ax in (self._ax_loss, self._ax_acc):
            if ep:
                ax.set_xlim(1, max(ep) + 0.5)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

        self._canvas.draw_idle()