"""
app/main_window.py
───────────────────
MainWindow: top-level application window.

Owns the QTabWidget, menu bar, and status bar.  Tabs are instantiated
lazily so the import of heavy widgets does not block application startup.
The status bar is wired to every tab's status_updated signal so any
background worker can push messages to a single, consistent location.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from core.registry import ModelRegistry


class MainWindow(QMainWindow):
    """
    Application shell.

    Parameters
    ----------
    registry : ModelRegistry
        Shared registry instance passed through to every tab that needs it.
    """

    WINDOW_TITLE   = "bnn-app"
    MIN_WIDTH      = 1200
    MIN_HEIGHT     = 780

    def __init__(self, registry: ModelRegistry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry

        self._setup_window()
        self._build_menu()
        self._build_tabs()
        self._build_status_bar()

        self.showMaximized()

    # ── Window setup ───────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

    # ── Tab widget ─────────────────────────────────────────────────────────────

    def _build_tabs(self) -> None:
        # Import here so startup is not blocked by heavy widget imports.
        from app.tabs.training_tab import TrainingTab
        from app.tabs.testing_tab import TestingTab
        from app.tabs.drawing_tab import DrawingTab

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(False)
        self._tabs.setMovable(False)
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)

        self._training_tab = TrainingTab(registry=self.registry)
        self._testing_tab  = TestingTab(registry=self.registry)
        self._drawing_tab  = DrawingTab(registry=self.registry)

        self._tabs.addTab(self._training_tab, "Training")
        self._tabs.addTab(self._testing_tab,  "Testing")
        self._tabs.addTab(self._drawing_tab,  "Drawing")

        # Wire each tab's status signal to the shared status bar.
        for tab in (self._training_tab, self._testing_tab, self._drawing_tab):
            if hasattr(tab, "status_message"):
                tab.status_message.connect(self._on_status)

        # When a model is saved in TrainingTab, refresh the model selectors
        # in TestingTab and DrawingTab so the new entry appears immediately.
        self._training_tab.training_completed.connect(
            lambda _: self._testing_tab.refresh_models()
        )
        self._training_tab.training_completed.connect(
            lambda _: self._drawing_tab.refresh_models()
        )

        self.setCentralWidget(self._tabs)

    # ── Menu bar ───────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu = self.menuBar()

        # ── File ──────────────────────────────────────────────────────────────
        file_menu = menu.addMenu("File")

        open_models = QAction("Open Models Directory", self)
        open_models.triggered.connect(self._open_models_dir)
        file_menu.addAction(open_models)

        open_exports = QAction("Open Exports Directory", self)
        open_exports.triggered.connect(self._open_exports_dir)
        file_menu.addAction(open_exports)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ── View ──────────────────────────────────────────────────────────────
        view_menu = menu.addMenu("View")

        for i, label in enumerate(("Training", "Testing", "Drawing")):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            # Capture i by default argument to avoid closure-over-loop pitfall
            action.triggered.connect(lambda _, idx=i: self._tabs.setCurrentIndex(idx))
            view_menu.addAction(action)

        # ── Help ──────────────────────────────────────────────────────────────
        help_menu = menu.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self._status_label = QLabel("Ready")
        bar.addWidget(self._status_label, 1)

        # Right-aligned indicator: which device PyTorch is using.
        device_text = self._detect_device_label()
        device_label = QLabel(device_text)
        device_label.setProperty("class", "status-ok" if "CUDA" in device_text else "metric-label")
        bar.addPermanentWidget(device_label)

    def _on_status(self, message: str) -> None:
        self._status_label.setText(message)

    @staticmethod
    def _detect_device_label() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return f"CUDA · {name}"
            return "CPU"
        except Exception:
            return "CPU"

    # ── Actions ────────────────────────────────────────────────────────────────

    def _open_models_dir(self) -> None:
        from core.config import SAVED_MODELS_DIR
        import os
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(SAVED_MODELS_DIR))   # Windows

    def _open_exports_dir(self) -> None:
        from core.config import EXPORTS_DIR
        import os
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(EXPORTS_DIR))         # Windows

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About bnn-app",
            "<b>bnn-app</b><br>"
            "Bayesian Neural Network experiments.<br><br>"
            "Compares deterministic and MC Dropout predictors<br>"
            "on MNIST in-distribution and OOD datasets.<br><br>"
            "<small>Built with PyTorch + PyQt6</small>",
        )

    # ── Close guard ────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Confirm close if a training run is in progress."""
        training_active = (
            hasattr(self, "_training_tab")
            and self._training_tab.is_training_active()
        )
        if training_active:
            reply = QMessageBox.question(
                self,
                "Training in progress",
                "A training run is active. Stop and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._training_tab.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()