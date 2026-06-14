"""
app/widgets/model_selector_widget.py
──────────────────────────────────────
ModelSelectorWidget: reusable model picker shared by TestingTab and DrawingTab.

Shows a labelled QComboBox populated from the ModelRegistry,
and a metadata panel that displays the selected model's key config
values (dropout rate, epochs, val accuracy, timestamp).

Signals
───────
model_selected(ModelEntry)
    Emitted whenever the active selection changes to a valid entry.
    If the registry is empty the signal is not emitted.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from core.registry import ModelEntry, ModelRegistry


class ModelSelectorWidget(QWidget):
    """
    Compact model picker with full training-parameter metadata panel.

    Parameters
    ----------
    registry : ModelRegistry
    parent   : QWidget, optional
    """

    model_selected = pyqtSignal(object)   # ModelEntry

    def __init__(
        self,
        registry: ModelRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry
        self._entries: list[ModelEntry] = []
        self._build_ui()
        self.refresh()

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload entries from the registry and repopulate the combo box."""
        self._entries = self.registry.list_models()
        self._combo.blockSignals(True)
        self._combo.clear()

        if not self._entries:
            self._combo.addItem("— no saved models —")
            self._combo.setEnabled(False)
            self._clear_meta()
        else:
            self._combo.setEnabled(True)
            for entry in self._entries:
                date = entry.timestamp[:10]
                self._combo.addItem(f"{entry.name}  ({date})")

        self._combo.blockSignals(False)
        self._combo.setCurrentIndex(0)
        self._on_selection_changed(0)

    def current_entry(self) -> Optional[ModelEntry]:
        idx = self._combo.currentIndex()
        if 0 <= idx < len(self._entries):
            return self._entries[idx]
        return None

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # Combo box — no refresh button; parent calls refresh() when needed.
        self._combo = QComboBox()
        self._combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        root.addWidget(self._combo)

        # Model info group — full training parameters
        self._meta_grp = QGroupBox("Model Info")
        form = QFormLayout(self._meta_grp)
        form.setContentsMargins(10, 8, 10, 8)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def _val() -> QLabel:
            lbl = QLabel("—")
            lbl.setStyleSheet(
                "font-family: 'Consolas', monospace; font-size: 11px;"
                "color: #f5a623; background: transparent;"
            )
            return lbl

        def _key(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-size: 11px; color: #555d6e; background: transparent;"
            )
            return lbl

        # One value label per field — stored in a dict for easy update.
        self._meta: dict[str, QLabel] = {}

        rows = [
            ("Dataset",      "train_dataset"),
            ("Train size",   "train_size"),
            ("Val split",    "val_split"),
            ("Epochs",       "epochs"),
            ("LR",           "lr"),
            ("Batch size",   "batch_size"),
            ("Weight decay", "weight_decay"),
            ("Dropout  p",   "dropout_rate"),
            ("Val acc",      "val_acc"),       # from training_metrics
            ("Device",       "device"),
            ("Saved",        "saved"),
        ]

        for display, key in rows:
            val_lbl = _val()
            form.addRow(_key(display), val_lbl)
            self._meta[key] = val_lbl

        root.addWidget(self._meta_grp)

    # ── Handlers ───────────────────────────────────────────────────────────────

    def _on_selection_changed(self, index: int) -> None:
        if 0 <= index < len(self._entries):
            entry = self._entries[index]
            self._populate_meta(entry)
            self.model_selected.emit(entry)
        else:
            self._clear_meta()

    def _populate_meta(self, entry: ModelEntry) -> None:
        cfg   = entry.config
        t     = cfg.get("training", {})
        mets  = entry.training_metrics

        def _fmt(v, decimals: int = 5) -> str:
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{decimals}f}".rstrip("0").rstrip(".")
            return str(v)

        self._meta["train_dataset"].setText(t.get("train_dataset", "—"))
        self._meta["train_size"].setText(_fmt(t.get("train_size"), 2))
        self._meta["val_split"].setText(_fmt(t.get("val_split"), 2))
        self._meta["epochs"].setText(str(t.get("epochs", "—")))
        self._meta["lr"].setText(_fmt(t.get("lr"), 5))
        self._meta["batch_size"].setText(str(t.get("batch_size", "—")))
        self._meta["weight_decay"].setText(_fmt(t.get("weight_decay"), 5))
        self._meta["dropout_rate"].setText(_fmt(t.get("dropout_rate"), 2))

        val_acc = mets.get("val_acc")
        self._meta["val_acc"].setText(
            f"{val_acc:.4f}" if val_acc is not None else "—"
        )
        self._meta["device"].setText(cfg.get("device") or "—")
        self._meta["saved"].setText(entry.timestamp[:10])

    def _clear_meta(self) -> None:
        for lbl in self._meta.values():
            lbl.setText("—")