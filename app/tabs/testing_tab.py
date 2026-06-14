"""
app/tabs/testing_tab.py
────────────────────────
TestingTab: evaluate a saved model on in-distribution and OOD datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.config import EXPORTS_DIR, AVAILABLE_DATASETS
from core.data.datasets import TRAIN_DATASETS, is_downloaded
from core.models.alexnet import AlexNetSmall
from core.registry import ModelEntry, ModelRegistry
from core.evaluation.evaluator import EvalResults
from app.workers.inference_worker import InferenceWorker, InferenceMode

_BG      = "#0d0f12"
_SURFACE = "#161920"
_GRID    = "#2a2e38"
_DIM     = "#555d6e"
_SEC     = "#8b92a5"
_AMBER   = "#f5a623"
_GREEN   = "#4caf7d"
_RED     = "#e05c5c"
_YELLOW  = "#e8a838"
_WHITE   = "#e8eaed"
_TEAL    = "#87ceab"
_BLUE    = "#4C9DAF"


class TestingTab(QWidget):
    status_message = pyqtSignal(str)

    def __init__(
        self,
        registry: ModelRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry
        self._worker: Optional[InferenceWorker] = None
        self._det_results:  Optional[EvalResults] = None
        self._mc_results:   Optional[EvalResults] = None
        self._active_entry: Optional[ModelEntry]  = None
        self._train_dataset: str                  = ""
        self._download_workers: dict[str, object] = {}
        self._total_ds: int = 0
        self._build_ui()
        self._auto_select_model()

    def refresh_models(self) -> None:
        self._model_selector.refresh()
        self._auto_select_model()

    def _auto_select_model(self) -> None:
        """Select and load the newest model automatically, mirroring DrawingTab."""
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
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root.addWidget(splitter)

    # ── Left panel ─────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        from app.widgets.model_selector_widget import ModelSelectorWidget

        container = QWidget()
        container.setFixedWidth(300)
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

        # Model selector
        self._model_selector = ModelSelectorWidget(self.registry)
        self._model_selector.model_selected.connect(self._on_model_selected)
        layout.addWidget(self._model_selector)

        # Testing datasets
        layout.addWidget(self._build_dataset_panel())

        # Inference mode — controls what is RUN (display is handled at top of right panel)
        mode_grp = QGroupBox("Inference Mode")
        mode_layout = QVBoxLayout(mode_grp)
        mode_layout.setSpacing(6)
        self._mode_group = QButtonGroup(self)
        self._rb_det  = QRadioButton("Deterministic")
        self._rb_mc   = QRadioButton("MC Dropout")
        self._rb_both = QRadioButton("Both")
        self._rb_both.setChecked(True)
        for i, rb in enumerate((self._rb_det, self._rb_mc, self._rb_both)):
            self._mode_group.addButton(rb, i)
            mode_layout.addWidget(rb)
        layout.addWidget(mode_grp)

        # Progress
        self._eval_progress = QProgressBar()
        self._eval_progress.setVisible(False)
        layout.addWidget(self._eval_progress)

        # Run button
        self._run_btn = QPushButton("▶  Run Evaluation")
        self._run_btn.setMinimumHeight(38)
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5a623; color: #0d0f12;
                border: none; border-radius: 3px;
                font-size: 13px; font-weight: 700;
                min-height: 38px; padding: 0 18px;
            }
            QPushButton:hover   { background-color: #fbb93d; color: #0d0f12; }
            QPushButton:pressed { background-color: #e09520; }
            QPushButton:disabled { background-color: #3d2e0d; color: #6b5520; }
        """)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        layout.addStretch()
        scroll.setWidget(inner)

        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(scroll)
        return container

    # ── Dataset panel ──────────────────────────────────────────────────────────

    def _build_dataset_panel(self) -> QGroupBox:
        grp = QGroupBox("Testing Datasets")
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)
        self._ds_rows: dict[str, dict] = {}

        for name in AVAILABLE_DATASETS:
            info       = TRAIN_DATASETS[name]
            downloaded = is_downloaded(name)

            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)

            cb = QCheckBox(info.display_name)
            cb.setChecked(downloaded)
            cb.setEnabled(downloaded)
            row_l.addWidget(cb, stretch=1)

            if downloaded:
                tag = QLabel("")
                tag.setStyleSheet(
                    "font-size: 10px; color: #555d6e; background: transparent;"
                )
                row_l.addWidget(tag)
                dl_btn = None
            else:
                tag    = None
                dl_btn = QPushButton("Download")
                dl_btn.setFixedHeight(22)
                dl_btn.setStyleSheet(
                    "font-size: 10px; padding: 1px 6px; min-height: 22px;"
                    "background-color: #f5a623; color: #0d0f12;"
                    "font-weight: 700; border: none; border-radius: 3px;"
                )
                dl_btn.clicked.connect(
                    lambda checked, n=name: self._on_download(n)
                )
                row_l.addWidget(dl_btn)

            layout.addWidget(row_w)
            self._ds_rows[name] = {"cb": cb, "dl_btn": dl_btn, "tag": tag}

        return grp

    # ── Right panel ────────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(10)

        # ── Display predictor selector (top of right panel) ───────────────────
        selector_row = QHBoxLayout()

        show_lbl = QLabel("Show:")
        show_lbl.setStyleSheet("font-weight: 700; color: #8b92a5;")
        selector_row.addWidget(show_lbl)

        self._display_group = QButtonGroup(self)
        self._rb_show_det = QRadioButton("Deterministic")
        self._rb_show_mc  = QRadioButton("MC Dropout")
        self._rb_show_det.setChecked(True)
        self._rb_show_det.setEnabled(False)
        self._rb_show_mc.setEnabled(False)

        for i, rb in enumerate((self._rb_show_det, self._rb_show_mc)):
            self._display_group.addButton(rb, i)
            rb.toggled.connect(self._on_display_changed)
            selector_row.addWidget(rb)

        selector_row.addStretch()
        layout.addLayout(selector_row)

        # Thin separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2a2e38;")
        layout.addWidget(sep)

        # ── Metric cards ──────────────────────────────────────────────────────
        layout.addWidget(self._build_metric_cards())

        # ── Result tabs ───────────────────────────────────────────────────────
        self._result_tabs = QTabWidget()
        self._result_tabs.addTab(self._build_ood_table_tab(),   "OOD Detection")
        self._result_tabs.addTab(self._build_entropy_tab(),     "Uncertainty")
        self._result_tabs.addTab(self._build_auroc_tab(),       "AUROC Curves")
        layout.addWidget(self._result_tabs, stretch=1)

        # Export
        export_row = QHBoxLayout()
        export_row.addStretch()
        self._export_btn = QPushButton("⬇  Export All Figures")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        export_row.addWidget(self._export_btn)
        layout.addLayout(export_row)

        return panel

    def _build_metric_cards(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._card_labels: dict[str, QLabel] = {}

        for title, key in (
            ("Accuracy", "acc"), ("F1 (macro)", "f1"),
            ("ECE",      "ece"),
        ):
            card = QFrame()
            card.setStyleSheet(
                "background-color: #161920; border: 1px solid #2a2e38;"
                " border-radius: 4px;"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 12, 12, 12)
            cl.setSpacing(4)

            val = QLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(
                "font-family:'Consolas',monospace; font-size:20px;"
                "font-weight:700; color:#f5a623;"
                "background:transparent; border:none;"
            )
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "font-size:11px; color:#555d6e;"
                "background:transparent; border:none;"
            )
            cl.addWidget(val)
            cl.addWidget(lbl)
            layout.addWidget(card, stretch=1)
            self._card_labels[key] = val

        return row

    def _build_ood_table_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        cols = ["Dataset", "AUROC ↑", "FPR95 ↓"]
        self._ood_table = QTableWidget(0, len(cols))
        self._ood_table.setHorizontalHeaderLabels(cols)
        self._ood_table.horizontalHeader().setStretchLastSection(False)
        self._ood_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._ood_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._ood_table.setAlternatingRowColors(True)
        self._ood_table.verticalHeader().setVisible(False)
        layout.addWidget(self._ood_table)
        return w

    def _build_entropy_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        self._ent_fig    = Figure(figsize=(8, 4), facecolor=_BG, tight_layout=True)
        self._ent_canvas = FigureCanvasQTAgg(self._ent_fig)
        self._ent_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._ent_canvas)
        return w

    def _build_auroc_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        self._auroc_fig    = Figure(figsize=(8, 4), facecolor=_BG, tight_layout=True)
        self._auroc_canvas = FigureCanvasQTAgg(self._auroc_fig)
        self._auroc_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._auroc_canvas)
        return w

    # ── Active result helper ───────────────────────────────────────────────────

    def _active_result(self) -> Optional[EvalResults]:
        """Return the result for whichever predictor is selected for display."""
        if self._display_group.checkedId() == 1 and self._mc_results is not None:
            return self._mc_results
        if self._det_results is not None:
            return self._det_results
        return self._mc_results   # fallback

    # ── Model selection ────────────────────────────────────────────────────────

    def _on_model_selected(self, entry: ModelEntry) -> None:
        self._active_entry   = entry
        self._det_results    = None
        self._mc_results     = None
        self._train_dataset  = entry.config.get("training", {}).get("train_dataset", "")
        dropout_rate         = entry.config.get("training", {}).get("dropout_rate", 0.0)

        # Reset display selector
        self._rb_show_det.setEnabled(False)
        self._rb_show_mc.setEnabled(False)
        self._rb_show_det.setChecked(True)

        # Update dataset tags
        for name, row in self._ds_rows.items():
            if row["tag"] is not None:
                if name == self._train_dataset:
                    row["tag"].setText("in-dist")
                    row["tag"].setStyleSheet(
                        "font-size:10px; color:#4caf7d; background:transparent;"
                    )
                else:
                    row["tag"].setText("OOD")
                    row["tag"].setStyleSheet(
                        "font-size:10px; color:#8b92a5; background:transparent;"
                    )

        # MC Dropout guard
        mc_ok = dropout_rate > 0.0
        self._rb_mc.setEnabled(mc_ok)
        self._rb_both.setEnabled(mc_ok)
        tip = "" if mc_ok else "MC Dropout unavailable: model trained with dropout_rate = 0."
        self._rb_mc.setToolTip(tip)
        self._rb_both.setToolTip(tip)
        if not mc_ok:
            self._rb_det.setChecked(True)
        else:
            self._rb_both.setChecked(True)

        self._run_btn.setEnabled(True)
        self._reset_display()

    # ── Display selector ───────────────────────────────────────────────────────

    def _on_display_changed(self) -> None:
        """Called when the user switches the Show: radio at top of right panel."""
        self._refresh_display()

    # ── Run ────────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        if not self._active_entry:
            return

        from core.data.datasets import DataManager

        entry  = self._active_entry
        config = self.registry.load_config(entry)
        config.inference.ood_datasets = [
            name for name, row in self._ds_rows.items()
            if row["cb"].isChecked() and name != self._train_dataset
        ]

        run_in_dist = self._ds_rows.get(
            self._train_dataset, {}
        ).get("cb", None)
        run_in_dist = run_in_dist is not None and run_in_dist.isChecked()

        model = AlexNetSmall(dropout_rate=config.training.dropout_rate)
        model.load_state_dict(self.registry.load_state_dict(entry))
        model.eval()

        data_mgr    = DataManager(config)
        test_loader = data_mgr.get_test_loader() if run_in_dist else None
        ood_loaders = data_mgr.get_ood_loaders()

        # ── Channel duplication ───────────────────────────────────────────────
        # DataManager yields single-channel (1, H, W) tensors; AlexNetSmall
        # expects 3-channel input.  Wrap every loader so batches are expanded
        # to (B, 3, H, W) before reaching the model — identical to the
        # .repeat(1, 3, 1, 1) call in DrawingTab._run_inference.
        if test_loader is not None:
            test_loader = _ChannelRepeatLoader(test_loader, n=3)
        ood_loaders = {name: _ChannelRepeatLoader(ldr, n=3)
                       for name, ldr in ood_loaders.items()}

        effective_test = test_loader or next(iter(ood_loaders.values()), None)
        effective_ood  = ood_loaders if test_loader else {}

        if effective_test is None:
            QMessageBox.warning(self, "Nothing Selected",
                                "Please select at least one dataset.")
            return

        mode_map = {
            0: InferenceMode.DETERMINISTIC,
            1: InferenceMode.MC_DROPOUT,
            2: InferenceMode.BOTH,
        }
        mode     = mode_map[self._mode_group.checkedId()]
        total_ds = (1 if run_in_dist else 0) + len(ood_loaders)

        # ── Cancel any in-flight worker before starting a new one ─────────────
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
            self._worker = None

        # For BOTH mode the deterministic pass uses a normal progress bar and
        # the MC pass switches to indeterminate (pulse) via _on_status_updated,
        # so seed the maximum with the per-pass dataset count.
        # For MC-only mode the bar goes straight to indeterminate once the
        # status signal fires, so any non-zero initial maximum is fine.
        self._total_ds = total_ds
        self._eval_progress.setMaximum(total_ds)
        self._eval_progress.setValue(0)
        self._eval_progress.setVisible(True)
        self._run_btn.setEnabled(False)
        self._export_btn.setEnabled(False)

        # Reset display radios for the new run
        self._rb_show_det.setEnabled(False)
        self._rb_show_mc.setEnabled(False)
        self._det_results = None
        self._mc_results  = None

        self._worker = InferenceWorker(
            config      = config,
            model       = model,
            test_loader = effective_test,
            ood_loaders = effective_ood,
            mode        = mode,
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.progress_updated.connect(
            lambda s, t: self._eval_progress.setValue(
                self._eval_progress.value() + 1
            )
        )
        self._worker.status_updated.connect(self._on_status_updated)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker.start()

    def _on_result(self, result: EvalResults) -> None:
        if "Deterministic" in result.predictor_name:
            self._det_results = result
            self._rb_show_det.setEnabled(True)
            self._rb_show_det.setChecked(True)   # auto-select as it arrives
        else:
            self._mc_results = result
            self._rb_show_mc.setEnabled(True)
            # Auto-select MC only if it's the only result available
            if self._det_results is None:
                self._rb_show_mc.setChecked(True)
        self._refresh_display()

    def _on_worker_done(self) -> None:
        self._eval_progress.setMaximum(1)   # exit indeterminate mode if still in it
        self._eval_progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(
            self._det_results is not None or self._mc_results is not None
        )
        self.status_message.emit("Evaluation complete.")

    def _on_status_updated(self, message: str) -> None:
        self.status_message.emit(message)
        # MC Dropout has no per-dataset ticks visible to the UI (each T-pass
        # batch is too slow for the progress signal to feel incremental).
        # Switch to indeterminate/pulse mode so the bar animates instead of
        # sitting frozen at 50%.  Deterministic mode restores the normal bar.
        if "MC Dropout" in message:
            self._eval_progress.setMaximum(0)   # pulse / indeterminate
        elif "Deterministic" in message:
            self._eval_progress.setMaximum(self._total_ds)
            self._eval_progress.setValue(0)

    def _on_error(self, tb: str) -> None:
        self._eval_progress.setVisible(False)
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Evaluation Error",
                             "An error occurred.\nCheck the terminal for details.")
        self.status_message.emit("Evaluation failed.")

    # ── Display ────────────────────────────────────────────────────────────────

    def _reset_display(self) -> None:
        for lbl in self._card_labels.values():
            lbl.setText("—")
        self._ood_table.setRowCount(0)
        for fig in (self._ent_fig, self._auroc_fig):
            fig.clear()
            fig.canvas.draw_idle()

    def _refresh_display(self) -> None:
        result = self._active_result()
        if result is None:
            return

        id_ = result.in_dist
        self._card_labels["acc"].setText(f"{id_.accuracy:.4f}")
        self._card_labels["f1"].setText(f"{id_.f1:.4f}")
        self._card_labels["ece"].setText(f"{id_.ece:.4f}")

        self._refresh_ood_table(result)
        self._draw_entropy(result)
        self._draw_auroc(result)

    def _refresh_ood_table(self, result: EvalResults) -> None:
        self._ood_table.setRowCount(0)
        for name, ood in result.ood.items():
            r = self._ood_table.rowCount()
            self._ood_table.insertRow(r)
            for c, val in enumerate([
                name.replace("_", " ").title(),
                f"{ood.auroc:.4f}",
                f"{ood.fpr95:.4f}",
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._ood_table.setItem(r, c, item)
        self._ood_table.resizeColumnsToContents()

    def _draw_entropy(self, result: EvalResults) -> None:
        self._ent_fig.clear()
        if result.in_dist_prediction is None:
            return

        ax = self._ent_fig.add_subplot(111)
        self._style_ax(ax)
        ax.hist(result.in_dist_prediction.entropy, bins=60,
                color=_GREEN, alpha=0.7, density=True, label="In-distribution")

        for (name, pred), color in zip(
            result.ood_predictions.items(), [_AMBER, _RED, _TEAL, _BLUE]
        ):
            ax.hist(pred.entropy, bins=60, color=color, alpha=0.55,
                    density=True, label=name.replace("_", " ").title())

        ax.set_xlabel("Predictive Entropy", color=_SEC, fontsize=10)
        ax.set_ylabel("Density",            color=_SEC, fontsize=10)
        ax.set_title(
            f"Entropy Distributions — {result.predictor_name}",
            color=_WHITE, fontsize=11
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, facecolor=_SURFACE,
                      edgecolor=_GRID, labelcolor=_SEC, fontsize=9)
        self._ent_canvas.draw_idle()

    def _draw_auroc(self, result: EvalResults) -> None:
        from sklearn.metrics import roc_curve

        self._auroc_fig.clear()
        if result.in_dist_prediction is None:
            return

        ax = self._auroc_fig.add_subplot(111)
        self._style_ax(ax)
        in_ent    = result.in_dist_prediction.entropy
        in_labels = np.zeros(len(in_ent))

        for (name, pred), color in zip(
            result.ood_predictions.items(), [_AMBER, _RED, _TEAL, _BLUE]
        ):
            scores = np.concatenate([in_ent, pred.entropy])
            labels = np.concatenate([in_labels, np.ones(len(pred.entropy))])
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = result.ood[name].auroc
            ax.plot(fpr, tpr, color=color, linewidth=1.8,
                    label=f"{name.replace('_', ' ').title()}  AUC={auc:.3f}")

        ax.plot([0, 1], [0, 1], color=_DIM, linewidth=1, linestyle=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("FPR", color=_SEC, fontsize=10)
        ax.set_ylabel("TPR", color=_SEC, fontsize=10)
        ax.set_title(
            f"ROC Curves — {result.predictor_name}",
            color=_WHITE, fontsize=11
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, facecolor=_SURFACE,
                      edgecolor=_GRID, labelcolor=_SEC, fontsize=9)
        self._auroc_canvas.draw_idle()

    # ── Download handlers ──────────────────────────────────────────────────────

    def _on_download(self, name: str) -> None:
        from app.workers.download_worker import DownloadWorker

        row = self._ds_rows[name]
        if row["dl_btn"]:
            row["dl_btn"].setEnabled(False)
            row["dl_btn"].setText("…")

        worker = DownloadWorker(name)
        worker.download_complete.connect(
            lambda n=name: self._on_download_done(n)
        )
        worker.error_occurred.connect(
            lambda tb, n=name: self._on_download_error(n, tb)
        )
        worker.status_updated.connect(self.status_message.emit)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(
            lambda n=name: self._download_workers.pop(n, None)
        )
        self._download_workers[name] = worker
        worker.start()

    def _on_download_done(self, name: str) -> None:
        row  = self._ds_rows[name]
        info = TRAIN_DATASETS[name]

        if row["dl_btn"]:
            row["dl_btn"].hide()
            row["dl_btn"].setParent(None)   # type: ignore[arg-type]
            row["dl_btn"].deleteLater()
            row["dl_btn"] = None

        tag = QLabel("")
        tag.setStyleSheet(
            "font-size:10px; color:#555d6e; background:transparent;"
        )
        row["cb"].parent().layout().addWidget(tag)
        row["tag"] = tag
        row["cb"].setEnabled(True)
        row["cb"].setChecked(True)

        if self._train_dataset == name:
            tag.setText("in-dist")
            tag.setStyleSheet(
                "font-size:10px; color:#4caf7d; background:transparent;"
            )
        elif self._train_dataset:
            tag.setText("OOD")

        self.status_message.emit(f"{info.display_name} downloaded successfully.")

    def _on_download_error(self, name: str, tb: str) -> None:
        row  = self._ds_rows[name]
        info = TRAIN_DATASETS.get(name)

        if row["dl_btn"]:
            row["dl_btn"].setEnabled(True)
            row["dl_btn"].setText("Download")

        QMessageBox.critical(
            self, f"Download Failed — {info.display_name if info else name}",
            f"Could not download.\n\nDetails:\n{tb[:400]}"
        )
        self.status_message.emit(f"Download failed: {name}.")

    # ── Export ─────────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select export directory", str(EXPORTS_DIR)
        )
        if not save_dir:
            return
        out   = Path(save_dir)
        saved = []
        for name, fig in (
            ("entropy",     self._ent_fig),
            ("auroc",       self._auroc_fig),
        ):
            if fig.get_axes():
                p = out / f"{name}.png"
                fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=_BG)
                saved.append(p.name)
        QMessageBox.information(
            self, "Export complete",
            f"Saved {len(saved)} figure(s) to:\n{save_dir}"
        )

    @staticmethod
    def _style_ax(ax) -> None:
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_DIM, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_color(_GRID)
        ax.spines["bottom"].set_color(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--", alpha=0.5)


# ── Utility ────────────────────────────────────────────────────────────────────

class _ChannelRepeatLoader:
    """
    Thin wrapper around a DataLoader that expands single-channel image batches
    to n-channel by repeating along the channel dimension.

    DataManager yields (B, 1, H, W) tensors; AlexNetSmall expects (B, 3, H, W).
    This mirrors the ``tensor.repeat(1, 3, 1, 1)`` call used in
    DrawingTab._run_inference for the same reason.

    The wrapper is transparent to the Evaluator: it supports iteration and
    exposes ``dataset`` so len() / attribute access still works.
    """

    def __init__(self, loader, n: int = 3) -> None:
        self._loader = loader
        self._n      = n

    # Forward dataset attribute so Evaluator can call len(loader.dataset).
    @property
    def dataset(self):
        return self._loader.dataset

    def __iter__(self):
        for images, labels in self._loader:
            if images.shape[1] == 1 and self._n != 1:
                images = images.repeat(1, self._n, 1, 1)
            yield images, labels

    def __len__(self) -> int:
        return len(self._loader)