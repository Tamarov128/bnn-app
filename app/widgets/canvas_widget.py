"""
app/widgets/canvas_widget.py
──────────────────────────────
CanvasWidget: a 28x28 MNIST-compatible drawing canvas.

The widget maintains an internal QImage at DISPLAY_SIZE x DISPLAY_SIZE
(280 x 280 by default — 10x the MNIST resolution) and renders brush
strokes with smooth anti-aliased lines.  On every mouse release it
downsamples the image to 28x28, normalises with MNIST statistics, and
emits canvas_changed with the resulting tensor.

Design decisions
────────────────
- Drawing convention: white strokes on black background, matching MNIST.
- Brush tool draws filled circles along the stroke path for smooth, gap-free lines.
- Undo stack (max 20 states) stored as QImage snapshots.
- get_tensor() and get_pil_image() allow both inference and gallery saving.
- A subtle grid overlay shows the 28-pixel cell boundaries when enabled.

Signals
───────
canvas_changed(torch.Tensor)
    Emitted on mouseReleaseEvent with a (1, 1, 28, 28) float32 tensor,
    normalised with MNIST mean=0.1307 and std=0.3081, ready for inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from PIL import Image as PILImage

from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QSize
from PyQt6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QBrush,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

# ── MNIST normalisation constants ──────────────────────────────────────────────
_MNIST_MEAN = 0.1307
_MNIST_STD  = 0.3081

# ── Canvas geometry ────────────────────────────────────────────────────────────
MNIST_SIZE   = 28          # target inference resolution
DISPLAY_SIZE = 280         # display resolution (10× MNIST)
SCALE        = DISPLAY_SIZE // MNIST_SIZE   # 10


class CanvasWidget(QWidget):
    """
    Interactive 28x28 MNIST drawing canvas.

    Parameters
    ----------
    brush_size : int
        Initial brush diameter in *display* pixels.
    show_grid  : bool
        Whether to render the 28-cell grid overlay.
    parent     : QWidget, optional
    """

    canvas_changed = pyqtSignal(object)   # torch.Tensor (1, 1, 28, 28)

    def __init__(
        self,
        brush_size: int  = 22,
        show_grid:  bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.brush_size = brush_size
        self.show_grid  = show_grid

        self._last_point: Optional[QPoint] = None
        self._drawing   = False
        self._undo_stack: list[QImage] = []

        self._init_canvas()

        self.setFixedSize(DISPLAY_SIZE, DISPLAY_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Draw a digit (0-9) here")

    # ── Public API ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Fill the canvas with black and emit an empty tensor."""
        self._push_undo()
        self._canvas.fill(QColor(0, 0, 0))
        self.update()
        self._emit_tensor()

    def undo(self) -> None:
        """Restore the previous canvas state."""
        if self._undo_stack:
            self._canvas = self._undo_stack.pop()
            self.update()
            self._emit_tensor()

    def set_brush_size(self, size: int) -> None:
        self.brush_size = max(1, size)

    def set_show_grid(self, visible: bool) -> None:
        self.show_grid = visible
        self.update()

    def get_tensor(self) -> torch.Tensor:
        """
        Return a (1, 1, 28, 28) float32 tensor normalised with MNIST stats.
        """
        arr = self._to_numpy_28()          # (28, 28) uint8
        t   = torch.from_numpy(arr).float() / 255.0   # [0, 1]
        t   = (t - _MNIST_MEAN) / _MNIST_STD
        return t.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)

    def get_pil_image(self) -> PILImage.Image:
        """Return the canvas as a 28x28 greyscale PIL image (for saving)."""
        arr = self._to_numpy_28()
        return PILImage.fromarray(arr, mode="L")

    def get_display_pixmap(self) -> QPixmap:
        """Return the current canvas as a QPixmap at display resolution."""
        return QPixmap.fromImage(self._canvas)

    # ── Qt painting ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self._canvas)

        if self.show_grid:
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            for i in range(1, MNIST_SIZE):
                x = i * SCALE
                painter.drawLine(x, 0, x, DISPLAY_SIZE)
                painter.drawLine(0, x, DISPLAY_SIZE, x)

        painter.end()

    # ── Mouse events ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._push_undo()
            self._drawing    = True
            self._last_point = event.position().toPoint()
            self._draw_dot(self._last_point)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drawing and self._last_point is not None:
            current = event.position().toPoint()
            self._draw_line(self._last_point, current)
            self._last_point = current
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing    = False
            self._last_point = None
            self.update()
            self._emit_tensor()

    # ── Drawing primitives ─────────────────────────────────────────────────────

    def _draw_dot(self, point: QPoint) -> None:
        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        r = self.brush_size // 2
        painter.drawEllipse(point, r, r)
        painter.end()
        self.update()

    def _draw_line(self, p1: QPoint, p2: QPoint) -> None:
        """Draw a thick, anti-aliased stroke from p1 to p2."""
        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(self.brush_size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _init_canvas(self) -> None:
        self._canvas = QImage(DISPLAY_SIZE, DISPLAY_SIZE, QImage.Format.Format_Grayscale8)
        self._canvas.fill(QColor(0, 0, 0))

    def _push_undo(self) -> None:
        self._undo_stack.append(self._canvas.copy())
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)

    def _to_numpy_28(self) -> np.ndarray:
        """Downsample canvas to 28x28 numpy array via PIL bilinear resize."""
        # Convert QImage (Grayscale8) to bytes
        ptr   = self._canvas.bits()
        ptr.setsize(DISPLAY_SIZE * DISPLAY_SIZE)
        arr   = np.frombuffer(ptr, dtype=np.uint8).reshape(DISPLAY_SIZE, DISPLAY_SIZE).copy()
        pil   = PILImage.fromarray(arr, mode="L")
        small = pil.resize((MNIST_SIZE, MNIST_SIZE), PILImage.BILINEAR)
        return np.array(small, dtype=np.uint8)

    def _emit_tensor(self) -> None:
        self.canvas_changed.emit(self.get_tensor())