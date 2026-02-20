from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProgressIndicator(QWidget):
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._total = 0
        self._current = 0
        self._is_cancelled = False

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("Processing...")
        self._label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self._label)

        progress_layout = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                background-color: #1e1e1e;
                text-align: center;
                color: #cccccc;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
        """)
        progress_layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        progress_layout.addWidget(self._cancel_btn)

        layout.addLayout(progress_layout)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #808080; font-size: 10px;")
        layout.addWidget(self._status_label)

    def start(self, total: int, label: str = "Processing..."):
        self._total = total
        self._current = 0
        self._is_cancelled = False
        self._label.setText(label)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(0)
        self._cancel_btn.setEnabled(True)
        self.setVisible(True)

    def update(self, current: int, status: str = ""):
        self._current = current
        self._progress_bar.setValue(current)
        if status:
            self._status_label.setText(status)
        if self._total > 0:
            percent = int((current / self._total) * 100)
            self._progress_bar.setFormat(f"{percent}%")

    def increment(self, status: str = ""):
        self._current += 1
        self.update(self._current, status)

    def finish(self, message: str = "Complete"):
        self._progress_bar.setValue(self._total)
        self._progress_bar.setFormat(message)
        self._cancel_btn.setEnabled(False)

    def set_label(self, text: str):
        self._label.setText(text)

    def set_status(self, text: str):
        self._status_label.setText(text)

    def _on_cancel(self):
        self._is_cancelled = True
        self.cancelled.emit()

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def reset(self):
        self._is_cancelled = False
        self._current = 0
        self._total = 0
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("")
        self._status_label.setText("")


class StatusBar:
    def __init__(self, widget: QWidget):
        self._widget = widget
        self._label = QLabel("Ready")
        self._label.setStyleSheet("color: #808080; padding: 4px;")

    def show_message(self, message: str, timeout: int = 0):
        self._label.setText(message)

    def clear_message(self):
        self._label.setText("Ready")

    def set_info(self, text: str):
        self._label.setText(text)
        self._label.setStyleSheet("color: #4caf50; padding: 4px;")

    def set_warning(self, text: str):
        self._label.setText(text)
        self._label.setStyleSheet("color: #ff9800; padding: 4px;")

    def set_error(self, text: str):
        self._label.setText(text)
        self._label.setStyleSheet("color: #f44336; padding: 4px;")
