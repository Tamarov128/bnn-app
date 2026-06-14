"""
main.py
────────
Application entry point.

    python main.py

Sets up high-DPI rendering, loads the stylesheet, instantiates the shared
ModelRegistry, and launches MainWindow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from core.config import EXPORTS_DIR, SAVED_MODELS_DIR, DATASETS_DIR
from core.registry import ModelRegistry


def main() -> int:
    # ── High-DPI ───────────────────────────────────────────────────────────────
    # Must be set before QApplication is created.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("bnn-app")
    app.setApplicationVersion("0.1.0")

    # ── Default font ───────────────────────────────────────────────────────────
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # ── Stylesheet ─────────────────────────────────────────────────────────────
    qss_path = Path(__file__).parent / "assets" / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    else:
        print(f"[main] Warning: stylesheet not found at {qss_path}")

    # ── Output directories ─────────────────────────────────────────────────────
    for d in (SAVED_MODELS_DIR, EXPORTS_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Shared registry ────────────────────────────────────────────────────────
    registry = ModelRegistry()

    # ── Main window ────────────────────────────────────────────────────────────
    from app.main_window import MainWindow
    window = MainWindow(registry=registry)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())