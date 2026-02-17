from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StepIndicator(QWidget):
    stepChanged = Signal(int)  # noqa: N815

    def __init__(self, steps: list[str], parent=None):
        super().__init__(parent)
        self._steps = steps
        self._current_step = 0
        self._buttons: list[QPushButton] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for i, step in enumerate(self._steps):
            btn = QPushButton(f"{i + 1}. {step}")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setEnabled(False)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #888;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:checked {
                    background-color: #0e639c;
                    color: white;
                }
                QPushButton:enabled {
                    color: #ccc;
                }
            """)
            btn.clicked.connect(lambda checked, idx=i: self._on_step_click(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

    def _on_step_click(self, index: int):
        self.set_current_step(index)

    def set_current_step(self, index: int):
        if 0 <= index < len(self._steps):
            self._current_step = index
            for i, btn in enumerate(self._buttons):
                btn.setChecked(i == index)
            self.stepChanged.emit(index)

    def complete_step(self, index: int):
        if 0 <= index < len(self._buttons):
            self._buttons[index].setStyleSheet("""
                QPushButton {
                    background-color: #2d5a1e;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:checked {
                    background-color: #0e639c;
                }
            """)

    def enable_step(self, index: int, enabled: bool = True):
        if 0 <= index < len(self._buttons):
            self._buttons[index].setEnabled(enabled)
            self._buttons[index].setStyleSheet("" if enabled else """
                QPushButton:disabled {
                    background-color: #3c3c3c;
                    color: #666;
                }
            """)

    def current_step(self) -> int:
        return self._current_step


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = False
        self._content_widget: QWidget | None = None
        self._toggle_btn: QPushButton | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle_btn = QPushButton(f"▼ {self._title}")
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: #ccc;
                border: none;
                padding: 8px 12px;
                text-align: left;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
        """)
        self._toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self._toggle_btn)

    def _toggle(self):
        self._collapsed = not self._collapsed
        arrow = "▶" if self._collapsed else "▼"
        self._toggle_btn.setText(f"{arrow} {self._title}")
        if self._content_widget:
            self._content_widget.setVisible(not self._collapsed)

    def set_content(self, widget: QWidget):
        self._content_widget = widget
        self.layout().addWidget(widget)


class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = "idle"
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self._dot)

        self._text = QLabel("Ready")
        self._text.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._text)
        layout.addStretch()

    def set_status(self, status: str, message: str):
        self._status = status
        colors = {
            "idle": "#666",
            "loading": "#f0ad4e",
            "success": "#5cb85c",
            "error": "#d9534f",
        }
        self._dot.setStyleSheet(f"color: {colors.get(status, '#666')}; font-size: 14px;")
        self._text.setText(message)


class IconButton(QPushButton):
    def __init__(self, icon: str = "", tooltip: str = "", parent=None):
        super().__init__(icon, "", parent)
        if tooltip:
            self.setToolTip(tooltip)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ccc;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
            QPushButton:pressed {
                background-color: #4c4c4c;
            }
        """)
