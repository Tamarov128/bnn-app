"""
app/tabs/training_tab.py
─────────────────────────
Training configuration + live monitoring tab.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.config import (
    AVAILABLE_DATASETS,
    ExperimentConfig,
    InferenceConfig,
    TrainingConfig,
)
from core.data.datasets import TRAIN_DATASETS, is_downloaded
from core.models.alexnet import AlexNetSmall
from core.registry import ModelRegistry


class TrainingTab(QWidget):
    """
    Training configuration and live monitoring tab.

    Signals
    ───────
    status_message(str)       — forwarded to MainWindow status bar
    training_completed(str)   — emitted with registry name after save
    """

    status_message     = pyqtSignal(str)
    training_completed = pyqtSignal(str)

    def __init__(
        self,
        registry: ModelRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry
        self._worker  = None
        self._model   = None
        self._trained_state_dict = None
        self._final_metrics      = None
        self._config             = None
        self._download_workers: dict[str, object] = {}
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_training_active(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def stop_training(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root.addWidget(splitter)

    # ── Left panel ─────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(320)
        container.setObjectName("left_panel")
        container.setStyleSheet(
            "#left_panel { background-color: #161920;"
            " border-right: 1px solid #2a2e38; }"
        )

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        layout.addWidget(self._build_dataset_group())
        layout.addWidget(self._build_training_group())
        layout.addWidget(self._build_save_group())
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(scroll)
        return container

    # ── Dataset selector ───────────────────────────────────────────────────────

    def _build_dataset_group(self) -> QGroupBox:
        grp = QGroupBox("Training Dataset")
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        self._dataset_radio_group = QButtonGroup(self)
        self._dataset_rows: dict[str, dict] = {}

        first_enabled = -1

        for i, name in enumerate(AVAILABLE_DATASETS):
            info       = TRAIN_DATASETS[name]
            downloaded = is_downloaded(name)

            if downloaded and first_enabled < 0:
                first_enabled = i

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            radio = QRadioButton(info.display_name)
            radio.setEnabled(downloaded)
            self._dataset_radio_group.addButton(radio, i)
            row_layout.addWidget(radio, stretch=1)

            status = QLabel("✓" if downloaded else "")
            status.setFixedWidth(18)
            if downloaded:
                status.setStyleSheet("color: #4caf7d; font-weight: 700;")
            row_layout.addWidget(status)

            if downloaded:
                dl_btn = None
            else:
                dl_btn = QPushButton("Download")
                dl_btn.setFixedHeight(30)
                dl_btn.setStyleSheet(
                    "font-size: 11px; padding: 2px 8px; min-height: 24px;"
                    "background-color: #f5a623; color: #0d0f12;"
                    "font-weight: 700; border: none; border-radius: 3px;"
                )
                dl_btn.clicked.connect(
                    lambda checked, n=name: self._on_download(n)
                )
                row_layout.addWidget(dl_btn)

            layout.addWidget(row_widget)
            self._dataset_rows[name] = {
                "radio":  radio,
                "status": status,
                "dl_btn": dl_btn,
            }

        if first_enabled >= 0:
            self._dataset_radio_group.button(first_enabled).setChecked(True)

        return grp

    # ── Training group ─────────────────────────────────────────────────────────

    def _build_training_group(self) -> QGroupBox:
        grp = QGroupBox("Training")
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        def row(label: str, widget: QWidget) -> QHBoxLayout:
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(100)
            r.addWidget(lbl)
            r.addWidget(widget)
            return r

        # ── Data split (top) ──────────────────────────────────────────────────
        self._train_size_spin = QDoubleSpinBox()
        self._train_size_spin.setRange(0.01, 1.0)
        self._train_size_spin.setSingleStep(0.05)
        self._train_size_spin.setValue(1.0)
        self._train_size_spin.setDecimals(2)
        self._train_size_spin.setToolTip(
            "Fraction of the training set to use.\n"
            "Set < 1.0 for small-dataset experiments."
        )
        layout.addLayout(row("Train size", self._train_size_spin))

        self._val_split_spin = QDoubleSpinBox()
        self._val_split_spin.setRange(0.05, 0.40)
        self._val_split_spin.setSingleStep(0.01)
        self._val_split_spin.setValue(0.10)
        self._val_split_spin.setDecimals(2)
        self._val_split_spin.setToolTip(
            "Fraction of training data held out for validation."
        )
        layout.addLayout(row("Val split", self._val_split_spin))

        # ── Divider ───────────────────────────────────────────────────────────
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2a2e38;")
        layout.addWidget(sep)

        # ── Optimiser ─────────────────────────────────────────────────────────
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(20)
        layout.addLayout(row("Epochs", self._epochs_spin))

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-5, 1.0)
        self._lr_spin.setSingleStep(1e-4)
        self._lr_spin.setDecimals(5)
        self._lr_spin.setValue(1e-3)
        layout.addLayout(row("Learning rate", self._lr_spin))

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(8, 1024)
        self._batch_spin.setSingleStep(8)
        self._batch_spin.setValue(128)
        layout.addLayout(row("Batch size", self._batch_spin))

        self._wd_spin = QDoubleSpinBox()
        self._wd_spin.setRange(0.0, 0.1)
        self._wd_spin.setSingleStep(1e-4)
        self._wd_spin.setDecimals(5)
        self._wd_spin.setValue(1e-4)
        layout.addLayout(row("Weight decay", self._wd_spin))

        # ── Architecture ──────────────────────────────────────────────────────
        self._dropout_spin = QDoubleSpinBox()
        self._dropout_spin.setRange(0.0, 0.9)
        self._dropout_spin.setSingleStep(0.05)
        self._dropout_spin.setValue(0.5)
        self._dropout_spin.setDecimals(2)
        self._dropout_spin.setToolTip(
            "Bernoulli dropout probability.\n"
            "Applied during training (regularisation) and at inference\n"
            "time when using MC Dropout (uncertainty estimation).\n"
            "Set to 0 for a purely deterministic model."
        )
        layout.addLayout(row("Dropout rate  p", self._dropout_spin))

        return grp

    # ── Save group ─────────────────────────────────────────────────────────────

    def _build_save_group(self) -> QGroupBox:
        grp = QGroupBox("Save Model")
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Run name"))
        self._run_name_edit = QLineEdit("unnamed_run")
        self._run_name_edit.setPlaceholderText("e.g. baseline_v1")
        layout.addWidget(self._run_name_edit)

        self._save_btn = QPushButton("Save to Registry")
        self._save_btn.setProperty("class", "primary")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        layout.addWidget(self._save_btn)

        self._save_status = QLabel("")
        self._save_status.setWordWrap(True)
        layout.addWidget(self._save_status)

        return grp

    # ── Right panel ────────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        from app.widgets.live_plot_widget import LivePlotWidget

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        self._plot = LivePlotWidget()
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._plot, stretch=3)

        # Progress
        progress_grp = QGroupBox("Progress")
        pg = QVBoxLayout(progress_grp)
        pg.setSpacing(6)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Epoch"))
        self._epoch_bar = QProgressBar()
        self._epoch_bar.setFormat("%v / %m")
        self._epoch_bar.setValue(0)
        epoch_row.addWidget(self._epoch_bar)
        pg.addLayout(epoch_row)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch"))
        self._batch_bar = QProgressBar()
        self._batch_bar.setFormat("%p%")
        self._batch_bar.setValue(0)
        batch_row.addWidget(self._batch_bar)
        pg.addLayout(batch_row)

        layout.addWidget(progress_grp, stretch=0)

        # Log
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setFixedHeight(120)
        self._log.setPlaceholderText("Training log will appear here…")
        layout.addWidget(self._log, stretch=0)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._start_btn = QPushButton("▶  Start Training")
        self._start_btn.setProperty("class", "primary")
        self._start_btn.setMinimumHeight(38)
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setProperty("class", "danger")
        self._stop_btn.setMinimumHeight(38)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        layout.addLayout(btn_row)
        return panel

    # ── Config ─────────────────────────────────────────────────────────────────

    def _selected_dataset(self) -> str:
        btn_id = self._dataset_radio_group.checkedId()
        return AVAILABLE_DATASETS[btn_id] if btn_id >= 0 else "mnist"

    def _build_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            run_name  = self._run_name_edit.text().strip() or "unnamed_run",
            training  = TrainingConfig(
                train_dataset = self._selected_dataset(),
                train_size    = self._train_size_spin.value(),
                val_split     = self._val_split_spin.value(),
                epochs        = self._epochs_spin.value(),
                lr            = self._lr_spin.value(),
                batch_size    = self._batch_spin.value(),
                weight_decay  = self._wd_spin.value(),
                dropout_rate  = self._dropout_spin.value(),
            ),
            inference = InferenceConfig(
                mc_samples   = 50,
                ood_datasets = AVAILABLE_DATASETS,
            ),
        )

    # ── Training handlers ──────────────────────────────────────────────────────

    def _on_start(self) -> None:
        from core.data.datasets import DataManager
        from app.workers.training_worker import TrainingWorker

        if self._dataset_radio_group.checkedId() < 0:
            QMessageBox.warning(
                self, "No Dataset",
                "Please download and select a training dataset first."
            )
            return

        self._config = self._build_config()
        self._model  = AlexNetSmall(
            dropout_rate = self._config.training.dropout_rate
        )

        try:
            dm           = DataManager(self._config)
            train_loader = dm.get_train_loader()
            val_loader   = dm.get_val_loader()
        except Exception as exc:
            QMessageBox.critical(self, "Data Error", str(exc))
            return

        self._epoch_bar.setMaximum(self._config.training.epochs)
        self._epoch_bar.setValue(0)
        self._batch_bar.setValue(0)
        self._log.clear()
        self._plot.reset()
        self._save_btn.setEnabled(False)
        self._save_status.setText("")
        self._trained_state_dict = None

        self._worker = TrainingWorker(
            config       = self._config,
            model        = self._model,
            train_loader = train_loader,
            val_loader   = val_loader,
        )
        self._worker.epoch_completed.connect(self._on_epoch)
        self._worker.batch_completed.connect(self._on_batch)
        self._worker.training_finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.status_updated.connect(self._on_status)
        self._worker.finished.connect(self._worker.deleteLater)

        self._set_running(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._log.appendPlainText("[Stopping after current batch…]")

    def _on_save(self) -> None:
        if not self._trained_state_dict or not self._config:
            return
        save_name = (
            self._run_name_edit.text().strip()
            or "unnamed_run"
        )
        self._config.run_name = save_name
        try:
            entry = self.registry.save_model(
                name             = save_name,
                state_dict       = self._trained_state_dict,
                config           = self._config,
                training_metrics = self._final_metrics,
            )
            self._save_status.setProperty("class", "status-ok")
            self._save_status.setText(f"✓ Saved as '{entry.name}'")
            self._save_status.style().unpolish(self._save_status)
            self._save_status.style().polish(self._save_status)
            self._save_btn.setEnabled(False)
            self.training_completed.emit(entry.name)
        except Exception as exc:
            self._save_status.setProperty("class", "status-error")
            self._save_status.setText(f"✗ {exc}")
            self._save_status.style().unpolish(self._save_status)
            self._save_status.style().polish(self._save_status)

    # ── Download handlers ──────────────────────────────────────────────────────

    def _on_download(self, name: str) -> None:
        from app.workers.download_worker import DownloadWorker

        row = self._dataset_rows[name]
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
        row  = self._dataset_rows[name]
        info = TRAIN_DATASETS[name]

        if row["dl_btn"]:
            row["dl_btn"].hide()
            row["dl_btn"].setParent(None)   # type: ignore[arg-type]
            row["dl_btn"].deleteLater()
            row["dl_btn"] = None

        row["status"].setText("✓")
        row["status"].setStyleSheet("color: #4caf7d; font-weight: 700;")
        row["radio"].setEnabled(True)

        if self._dataset_radio_group.checkedId() < 0:
            row["radio"].setChecked(True)

        self.status_message.emit(
            f"{info.display_name} downloaded successfully."
        )

    def _on_download_error(self, name: str, tb: str) -> None:
        row  = self._dataset_rows[name]
        info = TRAIN_DATASETS.get(name)
        display = info.display_name if info else name

        if row["dl_btn"]:
            row["dl_btn"].setEnabled(True)
            row["dl_btn"].setText("Download")

        QMessageBox.critical(
            self, f"Download Failed — {display}",
            f"Could not download {display}.\n\nDetails:\n{tb[:400]}"
        )
        self.status_message.emit(f"Download failed: {display}.")

    # ── Training signal handlers ────────────────────────────────────────────────

    def _on_epoch(
        self, epoch: int, train_loss: float, val_loss: float,
        train_acc: float, val_acc: float,
    ) -> None:
        self._epoch_bar.setValue(epoch + 1)
        self._plot.on_epoch(epoch, train_loss, val_loss, train_acc, val_acc)

    def _on_batch(
        self, epoch: int, batch_idx: int, total_batches: int, loss: float,
    ) -> None:
        self._batch_bar.setValue(int((batch_idx + 1) / total_batches * 100))

    def _on_finished(self, state_dict: dict, metrics: dict) -> None:
        self._trained_state_dict = state_dict
        self._final_metrics      = metrics
        self._set_running(False)
        self._save_btn.setEnabled(True)
        self._run_name_edit.setEnabled(True)
        self._batch_bar.setValue(100)
        

    def _on_error(self, tb: str) -> None:
        self._set_running(False)
        self._log.appendPlainText(f"\n[ERROR]\n{tb}")
        QMessageBox.critical(
            self, "Training Error",
            "An error occurred during training.\nSee the log for details.",
        )

    def _on_status(self, message: str) -> None:
        self._log.appendPlainText(message)
        self.status_message.emit(message)

    # ── State helpers ──────────────────────────────────────────────────────────

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        for w in self.findChildren(QSpinBox):
            w.setEnabled(not running)
        for w in self.findChildren(QDoubleSpinBox):
            w.setEnabled(not running)
        for w in self.findChildren(QLineEdit):
            w.setEnabled(not running)
        for row in self._dataset_rows.values():
            if row["dl_btn"] is not None:
                row["dl_btn"].setEnabled(not running)