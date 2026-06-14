"""
app/widgets/gallery_widget.py
──────────────────────────────
GalleryWidget: scrollable grid of saved drawing cards.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image as PILImage

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.gallery_registry import GalleryRecord, GalleryRegistry

THUMB_SIZE  = 70
CARD_HEIGHT = 96
N_COLS      = 3


class GalleryWidget(QWidget):

    entry_removed = pyqtSignal(str)   # entry_id

    def __init__(
        self,
        registry: GalleryRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._records:  list[GalleryRecord] = []
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_for_model(self, model_name: str) -> None:
        self._records = self._registry.load_for_model(model_name)
        self._rebuild()

    def add_record(self, record: GalleryRecord) -> None:
        self._records.insert(0, record)
        self._rebuild()

    def clear(self) -> None:
        self._records.clear()
        self._rebuild()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("Gallery")
        title.setProperty("class", "section-title")
        header.addWidget(title)
        header.addStretch()
        self._count_lbl = QLabel("0 samples")
        self._count_lbl.setStyleSheet(
            "font-size: 11px; color: #555d6e; background: transparent;"
        )
        header.addWidget(self._count_lbl)
        root.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._grid_w = QWidget()
        self._grid   = QGridLayout(self._grid_w)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setSpacing(6)
        self._scroll.setWidget(self._grid_w)
        root.addWidget(self._scroll)

        self._empty_lbl = QLabel(
            "No samples saved yet.\nDraw something and press Save."
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet(
            "font-size: 11px; color: #555d6e; background: transparent;"
        )
        root.addWidget(self._empty_lbl)

    def _rebuild(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        n = len(self._records)
        self._empty_lbl.setVisible(n == 0)
        self._scroll.setVisible(n > 0)
        self._count_lbl.setText(f"{n} sample{'s' if n != 1 else ''}")

        for idx, record in enumerate(self._records):
            card = self._make_card(record)
            self._grid.addWidget(card, idx // N_COLS, idx % N_COLS)

        # Pad last row
        if n % N_COLS:
            for col in range(n % N_COLS, N_COLS):
                sp = QWidget()
                sp.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
                )
                self._grid.addWidget(sp, n // N_COLS, col)

    # ── Card ───────────────────────────────────────────────────────────────────

    def _make_card(self, record: GalleryRecord) -> QFrame:
        card = QFrame()
        card.setFixedHeight(CARD_HEIGHT)
        card.setStyleSheet(
            "QFrame {"
            "  background-color: #161920;"
            "  border: 1px solid #2a2e38;"
            "  border-radius: 4px;"
            "}"
            "QFrame:hover { border-color: #f5a623; }"
        )

        outer = QHBoxLayout(card)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ── Thumbnail ─────────────────────────────────────────────────────────
        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setPixmap(self._record_to_pixmap(record))
        outer.addWidget(thumb)

        # ── Vertical divider ──────────────────────────────────────────────────
        outer.addWidget(_vdiv())

        # ── Det column ────────────────────────────────────────────────────────
        outer.addLayout(
            self._pred_col("Det", record.det.pred_class,
                           record.det.entropy),
            stretch=1,
        )

        # ── Vertical divider ──────────────────────────────────────────────────
        outer.addWidget(_vdiv())

        # ── MC column ─────────────────────────────────────────────────────────
        outer.addLayout(
            self._pred_col("MC Drop", record.mc.pred_class,
                           record.mc.entropy),
            stretch=1,
        )

        # ── Delete button (right-aligned, vertically centred) ─────────────────
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.setStyleSheet(
            "QPushButton {"
            "  font-size: 11px; font-weight: 700;"
            "  background: transparent;"
            "  color: #e05c5c;"
            "  border: 1px solid #6b2b2b;"
            "  border-radius: 3px;"
            "  padding: 5px;"
            "  min-height: 0px;"
            "}"
            "QPushButton:hover { background-color: #2e1515; }"
        )
        eid = record.entry_id
        del_btn.clicked.connect(lambda _, e=eid: self._delete(e))
        outer.addWidget(
            del_btn,
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )

        return card

    @staticmethod
    def _pred_col(header: str, pred_class: int, entropy: float) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(1)
        col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        def _lbl(text: str, style: str, monospace: bool = False) -> QLabel:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            lbl.setStyleSheet(style + " background: transparent; border: none;")
            if monospace:
                font = lbl.font()
                font.setFamily("Consolas")
                lbl.setFont(font)
            return lbl

        _s_hdr  = "font-size: 9px; color: #8b92a5;"
        _s_pred = "font-size: 18px; font-weight: 700; color: #f5a623;"
        _s_ent  = "font-size: 10px; color: #8b92a5;"

        col.addWidget(_lbl(header,               _s_hdr))
        col.addWidget(_lbl(str(pred_class),      _s_pred, monospace=True))
        col.addWidget(_lbl(f"Entropy={entropy:.3f}", _s_ent, monospace=True))

        return col

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_pixmap(record: GalleryRecord) -> QPixmap:
        if record.image_path.exists():
            arr = np.array(
                PILImage.open(record.image_path).resize(
                    (THUMB_SIZE, THUMB_SIZE), PILImage.NEAREST
                ),
                dtype=np.uint8,
            )
        else:
            arr = np.zeros((THUMB_SIZE, THUMB_SIZE), dtype=np.uint8)
        qimg = QImage(
            arr.data, THUMB_SIZE, THUMB_SIZE, THUMB_SIZE,
            QImage.Format.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimg)

    def _delete(self, entry_id: str) -> None:
        self._registry.delete(entry_id)
        self._records = [r for r in self._records if r.entry_id != entry_id]
        self._rebuild()
        self.entry_removed.emit(entry_id)


# ── Shared helper ──────────────────────────────────────────────────────────────

def _vdiv() -> QWidget:
    """1px vertical divider."""
    w = QWidget()
    w.setFixedWidth(1)
    w.setFixedHeight(CARD_HEIGHT - 16)
    w.setStyleSheet("background-color: #2a2e38;")
    return w