from __future__ import annotations

from typing import Any


class LoadingOverlay:
    def __init__(self, parent: Any):
        self.parent = parent
        self.widget: Any | None = None

    def show(self, message: str = "Loading...") -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

        self.widget = QWidget(self.parent)
        self.widget.setWindowFlags(
            Qt.WindowType.ToolTip |
            Qt.WindowType.FramelessWindowHint
        )
        self.widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)

        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(200)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.widget.setLayout(layout)

        self._center()
        self.widget.show()

    def _center(self) -> None:
        if not self.parent or not self.widget:
            return

        parent_geo = self.parent.geometry()
        widget_size = self.widget.sizeHint()

        x = parent_geo.x() + (parent_geo.width() - widget_size.width()) // 2
        y = parent_geo.y() + (parent_geo.height() - widget_size.height()) // 2

        self.widget.setGeometry(x, y, widget_size.width(), widget_size.height())

    def set_message(self, message: str) -> None:
        if self.widget:
            self.label.setText(message)

    def hide(self) -> None:
        if self.widget:
            self.widget.close()
            self.widget = None

    def is_visible(self) -> bool:
        return self.widget is not None and self.widget.isVisible()


class LoadingManager:
    _instance: LoadingOverlay | None = None

    def __new__(cls, parent: Any = None) -> LoadingOverlay:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, parent: Any = None):
        if not self._initialized:
            self.parent = parent
            self._overlay: LoadingOverlay | None = None
            self._initialized = True

    def show(self, message: str = "Loading...") -> None:
        if self._overlay is None and self.parent:
            self._overlay = LoadingOverlay(self.parent)
        if self._overlay:
            self._overlay.show(message)

    def hide(self) -> None:
        if self._overlay:
            self._overlay.hide()
            self._overlay = None

    def set_message(self, message: str) -> None:
        if self._overlay:
            self._overlay.set_message(message)

    def is_visible(self) -> bool:
        return self._overlay is not None and self._overlay.is_visible()
