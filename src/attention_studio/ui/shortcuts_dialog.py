from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from attention_studio.ui.shortcuts import KeyboardShortcut, get_shortcut_manager


class ShortcutsHelpDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Available Keyboard Shortcuts")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Shortcut", "Action", "Description"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        manager = get_shortcut_manager()
        shortcuts = manager.get_all_shortcuts()

        self.table.setRowCount(len(shortcuts))
        for i, shortcut in enumerate(shortcuts):
            shortcut_text = self._format_shortcut(shortcut)
            self.table.setItem(i, 0, QTableWidgetItem(shortcut_text))
            self.table.setItem(i, 1, QTableWidgetItem(shortcut.action))
            self.table.setItem(i, 2, QTableWidgetItem(shortcut.description))

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        search_label = QLabel("Search:")
        search_layout = QGridLayout()
        search_layout.addWidget(search_label, 0, 0)
        layout.addLayout(search_layout)

        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _format_shortcut(self, shortcut: KeyboardShortcut) -> str:
        parts = list(shortcut.modifiers)
        parts.append(shortcut.key)
        return "+".join(parts)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event and event.key() == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)


def show_shortcuts_dialog(parent: QWidget | None = None) -> None:
    dialog = ShortcutsHelpDialog(parent)
    dialog.exec()


def register_shortcut_handler(action: str, handler: Callable) -> None:
    manager = get_shortcut_manager()
    manager.register_handler(action, handler)
