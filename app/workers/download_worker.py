"""
app/workers/download_worker.py
────────────────────────────────
QThread that downloads a single dataset without blocking the GUI.

Signals
───────
  finished()         — download completed successfully
  error_occurred(str) — download failed; carries human-readable message
  status_updated(str) — free-text progress forwarded to the status bar
"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal


class DownloadWorker(QThread):
    """
    Downloads one dataset by calling core.data.datasets.download_dataset().

    Signals
    -------
    download_complete()
        Emitted when the download succeeds.  (NOT named 'finished' to
        avoid shadowing QThread.finished.)
    error_occurred(str)
        Emitted if an exception is raised; carries the formatted traceback.
    status_updated(str)
        Free-text progress message, forwarded to the status bar.
    """

    download_complete = pyqtSignal()          # success — custom name
    error_occurred    = pyqtSignal(str)       # failure + traceback
    status_updated    = pyqtSignal(str)       # progress text

    def __init__(self, dataset_name: str, parent=None) -> None:
        super().__init__(parent)
        self.dataset_name = dataset_name

    def run(self) -> None:
        try:
            from core.data.datasets import TRAIN_DATASETS, download_dataset
            info    = TRAIN_DATASETS.get(self.dataset_name)
            display = info.display_name if info else self.dataset_name

            self.status_updated.emit(f"Downloading {display}…")
            download_dataset(self.dataset_name)
            self.status_updated.emit(f"{display} downloaded successfully.")
            self.download_complete.emit()

        except Exception:
            self.error_occurred.emit(traceback.format_exc())
        # QThread.finished is emitted automatically by Qt after run() returns.