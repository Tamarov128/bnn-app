"""
app/tabs/drawing_tab.py
────────────────────────
DrawingTab: interactive inference with side-by-side comparison of
deterministic vs. MC Dropout on hand-drawn digits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.gallery_registry import GalleryRegistry, InferenceRecord
from core.registry import ModelEntry, ModelRegistry
from core.models.alexnet import AlexNetSmall
from app.workers.inference_worker import InferenceWorker, InferenceMode

_BG      = "#0d0f12"
_SURFACE = "#161920"
_GRID    = "#2a2e38"
_DIM     = "#555d6e"
_SEC     = "#8b92a5"
_AMBER   = "#f5a623"
_GREEN   = "#4caf7d"
_BLUE    = "#7eb6e8"
_WHITE   = "#e8eaed"

_MNIST_CLASSES = [str(i) for i in range(10)]
_CLASS_COLORS  = [
    "#f5a623", "#4caf7d", "#e05c5c", "#87ceab",
    "#e8a838", "#7eb6e8", "#c97dd4", "#f07c6e",
    "#59c3a8", "#a8c070",
]


class DrawingTab(QWidget):
    status_message = pyqtSignal(str)

    def __init__(
        self,
        registry: ModelRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.registry      = registry
        self._gallery_reg  = GalleryRegistry()
        self._model: Optional[AlexNetSmall] = None
        self._config       = None
        self._model_name   = ""
        self._det_result   = None
        self._mc_result    = None
        self._worker: Optional[InferenceWorker] = None

        self._infer_timer = QTimer()
        self._infer_timer.setSingleShot(True)
        self._infer_timer.setInterval(80)
        self._infer_timer.timeout.connect(self._run_inference)

        self._build_ui()
        self._auto_select_model()

    def refresh_models(self) -> None:
        self._model_selector.refresh()
        # Re-run auto-select so newly saved models are reflected
        self._auto_select_model()

    # ── Auto-select ────────────────────────────────────────────────────────────

    def _auto_select_model(self) -> None:
        """
        Select and load the newest model from the registry automatically.
        If no models exist, leave the placeholder visible.
        """
        entry = self._model_selector.current_entry()
        if entry is not None:
            self._on_model_selected(entry)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([360, 840])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root.addWidget(splitter)

    # ── Left panel ─────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        from app.widgets.model_selector_widget import ModelSelectorWidget
        from app.widgets.canvas_widget import CanvasWidget

        container = QWidget()
        container.setFixedWidth(360)
        container.setStyleSheet(
            "background-color: #161920; border-right: 1px solid #2a2e38;"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Model selector — signal connected BEFORE auto-select runs
        self._model_selector = ModelSelectorWidget(self.registry)
        self._model_selector.model_selected.connect(self._on_model_selected)
        layout.addWidget(self._model_selector)

        # Canvas
        canvas_grp = QGroupBox("Canvas  (28 x 28)")
        canvas_layout = QVBoxLayout(canvas_grp)
        canvas_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._canvas = CanvasWidget(brush_size=22)
        self._canvas.canvas_changed.connect(self._on_canvas_changed)
        canvas_layout.addWidget(
            self._canvas, alignment=Qt.AlignmentFlag.AlignHCenter
        )
        layout.addWidget(canvas_grp)

        # Tools
        tools_grp = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_grp)
        tools_layout.setSpacing(10)

        brush_row = QHBoxLayout()
        brush_lbl = QLabel("Brush size")
        brush_lbl.setFixedWidth(72)
        brush_row.addWidget(brush_lbl)
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(4, 60)
        self._brush_slider.setValue(22)
        self._brush_slider.valueChanged.connect(
            lambda v: self._canvas.set_brush_size(v)
        )
        brush_row.addWidget(self._brush_slider)
        self._brush_val_lbl = QLabel("22 px")
        self._brush_val_lbl.setFixedWidth(40)
        self._brush_val_lbl.setStyleSheet(
            "font-family:'Consolas'; font-size:11px; color:#8b92a5;"
        )
        self._brush_slider.valueChanged.connect(
            lambda v: self._brush_val_lbl.setText(f"{v} px")
        )
        brush_row.addWidget(self._brush_val_lbl)
        tools_layout.addLayout(brush_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._grid_cb = QCheckBox("Grid")
        self._grid_cb.setChecked(False)
        self._grid_cb.toggled.connect(self._canvas.set_show_grid)
        btn_row.addWidget(self._grid_cb)
        btn_row.addStretch()
        undo_btn = QPushButton("⟲ Undo")
        undo_btn.setFixedHeight(28)
        undo_btn.clicked.connect(self._canvas.undo)
        btn_row.addWidget(undo_btn)
        clear_btn = QPushButton("✕ Clear")
        clear_btn.setFixedHeight(28)
        clear_btn.setProperty("class", "danger")
        clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(clear_btn)
        tools_layout.addLayout(btn_row)
        layout.addWidget(tools_grp)

        self._save_btn = QPushButton("⬇  Save to Gallery")
        self._save_btn.setProperty("class", "primary")
        self._save_btn.setMinimumHeight(38)
        self._save_btn.setStyleSheet(
            "color:#8b92a5;"
        )
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        layout.addWidget(self._save_btn)

        layout.addStretch()
        scroll.setWidget(inner)

        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(scroll)
        return container

    # ── Right panel ────────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        from app.widgets.gallery_widget import GalleryWidget

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Inference area (stacked: placeholder vs. columns) ─────────────────
        self._infer_stack = QStackedWidget()

        # Page 0: placeholder
        placeholder = QWidget()
        ph_layout   = QVBoxLayout(placeholder)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ph_icon = QLabel("⬤")
        ph_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_icon.setStyleSheet("font-size: 32px; color: #2a2e38;")
        ph_layout.addWidget(ph_icon)

        ph_title = QLabel("No model loaded")
        ph_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_title.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #3a3f4d;"
        )
        ph_layout.addWidget(ph_title)

        ph_sub = QLabel(
            "Select a model from the left panel.\n"
            "The newest saved model will be loaded automatically."
        )
        ph_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_sub.setStyleSheet("font-size: 12px; color: #555d6e;")
        ph_layout.addWidget(ph_sub)

        self._infer_stack.addWidget(placeholder)   # index 0

        # Page 1: inference columns
        cols_widget = QWidget()
        cols_layout = QHBoxLayout(cols_widget)
        cols_layout.setContentsMargins(0, 0, 0, 0)
        cols_layout.setSpacing(0)

        self._build_inference_column(cols_layout, mode="det")
        div = QWidget()
        div.setFixedWidth(1)
        div.setStyleSheet("background-color: #2a2e38;")
        cols_layout.addWidget(div)
        self._build_inference_column(cols_layout, mode="mc")

        self._infer_stack.addWidget(cols_widget)   # index 1

        layout.addWidget(self._infer_stack, stretch=2)

        # ── Gallery ───────────────────────────────────────────────────────────
        self._gallery = GalleryWidget(registry=self._gallery_reg)
        layout.addWidget(self._gallery, stretch=3)

        return panel

    def _build_inference_column(
        self, parent_layout: QHBoxLayout, mode: str
    ) -> None:
        """
        Build one inference column in-place and store widget refs as
        self._det_* / self._mc_*.
        """
        col = QWidget()
        layout = QVBoxLayout(col)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Title label (updated when model loads)
        title_color = "#f5a623" if mode == "det" else "#4caf7d"
        title = QLabel("—")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {title_color};"
            "background: transparent;"
        )
        layout.addWidget(title)

        # Probability bar chart
        fig = Figure(
            figsize=(3, 2.8), facecolor=_BG,
            tight_layout={"pad": 0.4}
        )
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        ax = fig.add_subplot(111)
        self._style_prob_ax(ax)
        bars = ax.barh(
            _MNIST_CLASSES, np.zeros(10),
            color=_CLASS_COLORS, height=0.65,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", color=_SEC, fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=_GRID, linewidth=0.5,
                linestyle="--", alpha=0.5)
        layout.addWidget(canvas, stretch=1)

        # Uncertainty row — both modes show same three metrics
        unc_row = QHBoxLayout()
        unc_row.setSpacing(0)
        unc_labels: dict[str, QLabel] = {}

        i = 1
        (name, key) = ("Entropy", "entropy")
        cell = QVBoxLayout()
        cell.setSpacing(1)
        cell.setAlignment(Qt.AlignmentFlag.AlignTop)

        color = _BLUE if key == "mi" else _AMBER
        val = QLabel("—")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(
            f"font-family:'Consolas'; font-size:13px;"
            f"font-weight:700; color:{color};"
            "background:transparent; border:none;"
        )
        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "font-size:9px; color:#555d6e;"
            "background:transparent; border:none;"
        )
        cell.addWidget(val)
        cell.addWidget(lbl)
        unc_row.addLayout(cell)
        unc_labels[key] = val

        if i < 2:
            sep = QWidget()
            sep.setFixedWidth(1)
            sep.setFixedHeight(34)
            sep.setStyleSheet("background-color:#2a2e38;")
            unc_row.addWidget(sep)

        layout.addLayout(unc_row)
        parent_layout.addWidget(col, stretch=1)

        # Store refs
        p = f"_{mode}_"
        setattr(self, f"{p}title",  title)
        setattr(self, f"{p}fig",    fig)
        setattr(self, f"{p}canvas", canvas)
        setattr(self, f"{p}ax",     ax)
        setattr(self, f"{p}bars",   bars)
        setattr(self, f"{p}unc",    unc_labels)

    # ── Handlers ───────────────────────────────────────────────────────────────

    def _on_model_selected(self, entry: ModelEntry) -> None:
        self._config     = self.registry.load_config(entry)
        self._model_name = entry.name
        self._model      = AlexNetSmall(
            dropout_rate=self._config.training.dropout_rate
        )
        self._model.load_state_dict(self.registry.load_state_dict(entry))
        self._model.eval()

        T = self._config.inference.mc_samples
        p = self._config.training.dropout_rate

        # Update column titles
        self._det_title.setText("Deterministic")
        mc_title = f"MC Dropout  T={T}"
        if p == 0.0:
            mc_title += "  ⚠ p=0"
        self._mc_title.setText(mc_title)

        # Switch to inference columns (hide placeholder)
        self._infer_stack.setCurrentIndex(1)

        # Load persisted gallery for this model
        self._gallery.load_for_model(self._model_name)

        self.status_message.emit(
            f"Loaded: {entry.name}  (dropout p={p}, T={T})"
        )
        # Run inference on whatever is on the canvas right now
        if hasattr(self, "_pending_tensor"):
            self._run_inference()

    def _on_canvas_changed(self, tensor) -> None:
        self._pending_tensor = tensor
        if self._model is not None:
            self._infer_timer.start()

    def _on_clear(self) -> None:
        self._canvas.clear()
        self._reset_columns()

    def _on_save(self) -> None:
        if self._det_result is None or self._mc_result is None:
            return

        image_28 = np.array(self._canvas.get_pil_image(), dtype=np.uint8)
        det_r, mc_r = self._det_result, self._mc_result

        record = self._gallery_reg.save(
            model_name = self._model_name,
            image_28   = image_28,
            det = InferenceRecord(
                probs      = det_r["probs"].tolist(),
                entropy    = float(det_r["entropy"]),
                pred_class = int(np.argmax(det_r["probs"])),
            ),
            mc = InferenceRecord(
                probs      = mc_r["probs"].tolist(),
                entropy    = float(mc_r["entropy"]),
                pred_class = int(np.argmax(mc_r["probs"])),
            ),
        )
        self._gallery.add_record(record)
        self.status_message.emit(
            f"Saved — Det: {record.det.pred_class}  "
            f"MC: {record.mc.pred_class}"
        )

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_inference(self) -> None:
        if self._model is None or not hasattr(self, "_pending_tensor"):
            return

        # Cancel any in-flight worker before starting a new one.
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()

        # Reset results so _on_worker_result can use None as a first-pass sentinel.
        self._det_result = None
        self._mc_result  = None
        self._save_btn.setEnabled(False)

        # Replicate greyscale channel to satisfy AlexNetSmall's 3-channel input.
        tensor = self._pending_tensor.repeat(1, 3, 1, 1)  # (1,1,28,28) → (1,3,28,28)

        self._worker = InferenceWorker(
            config       = self._config,
            model        = self._model,
            single_image = tensor,
            mode         = InferenceMode.BOTH,
        )
        self._worker.single_result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_result(self, result) -> None:
        """
        Receives a PredictionResult (or _SingleMCResult) from
        InferenceWorker._run_single.

        InferenceMode.BOTH emits single_result_ready twice, in order:
          1st emission → deterministic  (_det_result is None sentinel)
          2nd emission → MC Dropout     (carries .mi for mutual information)
        """
        probs   = result.probs[0]        # (C,) — single sample
        entropy = float(result.entropy[0])

        if self._det_result is None:
            # ── First emission: deterministic ──────────────────────────────
            self._det_result = {"probs": probs, "entropy": entropy}
            self._update_column("det", probs, entropy)
        else:
            # ── Second emission: MC Dropout ────────────────────────────────
            mi = float(result.mi[0]) if hasattr(result, "mi") else 0.0
            self._mc_result = {"probs": probs, "entropy": entropy, "mi": mi}
            self._update_column("mc", probs, entropy, mi=mi)
            self._save_btn.setEnabled(True)

    def _on_worker_error(self, message: str) -> None:
        self.status_message.emit(f"Inference error: {message}")

    # ── Display ────────────────────────────────────────────────────────────────

    def _update_column(
        self,
        mode:    str,
        probs:   np.ndarray,
        entropy: float,
        *,
        mi:      float = 0.0,
    ) -> None:
        bars   = getattr(self, f"_{mode}_bars")
        canvas = getattr(self, f"_{mode}_canvas")
        unc    = getattr(self, f"_{mode}_unc")
        winner = int(np.argmax(probs))

        for i, (bar, p) in enumerate(zip(bars, probs)):
            bar.set_width(float(p))
            bar.set_edgecolor(_AMBER if i == winner else "none")
            bar.set_linewidth(1.5 if i == winner else 0)
            bar.set_alpha(1.0 if i == winner else 0.6)
        canvas.draw_idle()

        unc["entropy"].setText(f"{entropy:.4f}")
        # MI is only available for the MC column; the det column has no "mi" key.
        if "mi" in unc:
            unc["mi"].setText(f"{mi:.4f}")

    def _reset_columns(self) -> None:
        self._det_result = None
        self._mc_result  = None
        self._save_btn.setEnabled(False)
        for mode in ("det", "mc"):
            for bar in getattr(self, f"_{mode}_bars"):
                bar.set_width(0)
            getattr(self, f"_{mode}_canvas").draw_idle()
            for key, lbl in getattr(self, f"_{mode}_unc").items():
                if key != "mi" or mode == "mc":
                    lbl.setText("—")

    @staticmethod
    def _style_prob_ax(ax) -> None:
        ax.set_facecolor(_SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_SEC, labelsize=9)
        ax.xaxis.label.set_color(_SEC)


# ── Utility ────────────────────────────────────────────────────────────────────

def _entropy(probs: np.ndarray) -> float:
    c = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(c * np.log(c)))