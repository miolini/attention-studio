from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class ThemeColors:
    PRIMARY = "#0078D4"
    PRIMARY_HOVER = "#106EBE"
    PRIMARY_ACTIVE = "#005A9E"
    SECONDARY = "#6B6B6B"
    ACCENT = "#00B7C3"
    SUCCESS = "#107C10"
    WARNING = "#FF8C00"
    ERROR = "#D13438"
    INFO = "#0078D4"

    BACKGROUND_PRIMARY = "#FFFFFF"
    BACKGROUND_SECONDARY = "#F5F5F5"
    BACKGROUND_TERTIARY = "#E8E8E8"
    BACKGROUND_DARK = "#1E1E1E"
    BACKGROUND_DARK_SECONDARY = "#252526"
    BACKGROUND_DARK_TERTIARY = "#2D2D30"

    TEXT_PRIMARY = "#1A1A1A"
    TEXT_SECONDARY = "#6B6B6B"
    TEXT_DISABLED = "#A0A0A0"
    TEXT_ON_PRIMARY = "#FFFFFF"
    TEXT_DARK = "#FFFFFF"
    TEXT_DARK_SECONDARY = "#CCCCCC"

    BORDER = "#E0E0E0"
    BORDER_DARK = "#3F3F46"

    HOVER = "#F0F0F0"
    HOVER_DARK = "#3A3A3D"

    SELECTION = "#0078D4"
    SELECTION_DARK = "#264F78"


def _get_light_stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        color: {ThemeColors.TEXT_PRIMARY};
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 13px;
    }}
    QMenuBar {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        border-bottom: 1px solid {ThemeColors.BORDER};
        padding: 4px;
    }}
    QMenuBar::item {{
        padding: 6px 12px;
        background: transparent;
    }}
    QMenuBar::item:selected {{
        background-color: {ThemeColors.HOVER};
    }}
    QMenu {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        border: 1px solid {ThemeColors.BORDER};
    }}
    QMenu::item {{
        padding: 8px 24px;
    }}
    QMenu::item:selected {{
        background-color: {ThemeColors.HOVER};
    }}
    QToolBar {{
        background-color: {ThemeColors.BACKGROUND_SECONDARY};
        border: none;
        padding: 4px;
        spacing: 4px;
    }}
    QPushButton {{
        background-color: {ThemeColors.PRIMARY};
        color: {ThemeColors.TEXT_ON_PRIMARY};
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {ThemeColors.PRIMARY_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {ThemeColors.PRIMARY_ACTIVE};
    }}
    QPushButton:disabled {{
        background-color: {ThemeColors.BACKGROUND_TERTIARY};
        color: {ThemeColors.TEXT_DISABLED};
    }}
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        border: 1px solid {ThemeColors.BORDER};
        border-radius: 4px;
        padding: 8px;
        color: {ThemeColors.TEXT_PRIMARY};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {ThemeColors.PRIMARY};
    }}
    QTableWidget, QTreeWidget, QListWidget {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        alternate-background-color: {ThemeColors.BACKGROUND_SECONDARY};
        border: 1px solid {ThemeColors.BORDER};
        gridline-color: {ThemeColors.BORDER};
    }}
    QHeaderView::section {{
        background-color: {ThemeColors.BACKGROUND_SECONDARY};
        padding: 8px;
        border: none;
        border-bottom: 1px solid {ThemeColors.BORDER};
        font-weight: 600;
    }}
    QScrollBar:vertical {{
        background: {ThemeColors.BACKGROUND_SECONDARY};
        width: 12px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {ThemeColors.TEXT_SECONDARY};
        border-radius: 6px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ThemeColors.TEXT_PRIMARY};
    }}
    QScrollBar:horizontal {{
        background: {ThemeColors.BACKGROUND_SECONDARY};
        height: 12px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background: {ThemeColors.TEXT_SECONDARY};
        border-radius: 6px;
        min-width: 30px;
    }}
    QProgressBar {{
        border: 1px solid {ThemeColors.BORDER};
        border-radius: 4px;
        background-color: {ThemeColors.BACKGROUND_SECONDARY};
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {ThemeColors.PRIMARY};
        border-radius: 3px;
    }}
    QComboBox {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        border: 1px solid {ThemeColors.BORDER};
        border-radius: 4px;
        padding: 6px 12px;
    }}
    QComboBox::drop-down {{
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        border: 1px solid {ThemeColors.BORDER};
        selection-background-color: {ThemeColors.SELECTION};
    }}
    QTabWidget::pane {{
        border: 1px solid {ThemeColors.BORDER};
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
    }}
    QTabBar::tab {{
        background-color: {ThemeColors.BACKGROUND_SECONDARY};
        padding: 10px 20px;
        border: none;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        border-bottom: 2px solid {ThemeColors.PRIMARY};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {ThemeColors.BORDER};
        border-radius: 3px;
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ThemeColors.PRIMARY};
        border-color: {ThemeColors.PRIMARY};
    }}
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {ThemeColors.BORDER};
        border-radius: 9px;
    }}
    QRadioButton::indicator:checked {{
        background-color: {ThemeColors.PRIMARY};
        border-color: {ThemeColors.PRIMARY};
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {ThemeColors.BORDER};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 16px;
        margin: -6px 0;
        background: {ThemeColors.PRIMARY};
        border-radius: 8px;
    }}
    QSplitter::handle {{
        background-color: {ThemeColors.BORDER};
    }}
    QToolTip {{
        background-color: {ThemeColors.BACKGROUND_PRIMARY};
        color: {ThemeColors.TEXT_PRIMARY};
        border: 1px solid {ThemeColors.BORDER};
        padding: 4px 8px;
    }}
    """


def _get_dark_stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background-color: {ThemeColors.BACKGROUND_DARK};
        color: {ThemeColors.TEXT_DARK};
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 13px;
    }}
    QMenuBar {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        border-bottom: 1px solid {ThemeColors.BORDER_DARK};
        padding: 4px;
    }}
    QMenuBar::item {{
        padding: 6px 12px;
        background: transparent;
    }}
    QMenuBar::item:selected {{
        background-color: {ThemeColors.HOVER_DARK};
    }}
    QMenu {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        border: 1px solid {ThemeColors.BORDER_DARK};
    }}
    QMenu::item {{
        padding: 8px 24px;
    }}
    QMenu::item:selected {{
        background-color: {ThemeColors.HOVER_DARK};
    }}
    QToolBar {{
        background-color: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        border: none;
        padding: 4px;
        spacing: 4px;
    }}
    QPushButton {{
        background-color: {ThemeColors.PRIMARY};
        color: {ThemeColors.TEXT_ON_PRIMARY};
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {ThemeColors.PRIMARY_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {ThemeColors.PRIMARY_ACTIVE};
    }}
    QPushButton:disabled {{
        background-color: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        color: {ThemeColors.TEXT_DISABLED};
    }}
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        border: 1px solid {ThemeColors.BORDER_DARK};
        border-radius: 4px;
        padding: 8px;
        color: {ThemeColors.TEXT_DARK};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {ThemeColors.PRIMARY};
    }}
    QTableWidget, QTreeWidget, QListWidget {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        alternate-background-color: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        border: 1px solid {ThemeColors.BORDER_DARK};
        gridline-color: {ThemeColors.BORDER_DARK};
    }}
    QHeaderView::section {{
        background-color: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        padding: 8px;
        border: none;
        border-bottom: 1px solid {ThemeColors.BORDER_DARK};
        font-weight: 600;
    }}
    QScrollBar:vertical {{
        background: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        width: 12px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {ThemeColors.TEXT_DARK_SECONDARY};
        border-radius: 6px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ThemeColors.TEXT_DARK};
    }}
    QScrollBar:horizontal {{
        background: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        height: 12px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background: {ThemeColors.TEXT_DARK_SECONDARY};
        border-radius: 6px;
        min-width: 30px;
    }}
    QProgressBar {{
        border: 1px solid {ThemeColors.BORDER_DARK};
        border-radius: 4px;
        background-color: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {ThemeColors.PRIMARY};
        border-radius: 3px;
    }}
    QComboBox {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        border: 1px solid {ThemeColors.BORDER_DARK};
        border-radius: 4px;
        padding: 6px 12px;
    }}
    QComboBox::drop-down {{
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        border: 1px solid {ThemeColors.BORDER_DARK};
        selection-background-color: {ThemeColors.SELECTION_DARK};
    }}
    QTabWidget::pane {{
        border: 1px solid {ThemeColors.BORDER_DARK};
        background-color: {ThemeColors.BACKGROUND_DARK};
    }}
    QTabBar::tab {{
        background-color: {ThemeColors.BACKGROUND_DARK_TERTIARY};
        padding: 10px 20px;
        border: none;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        border-bottom: 2px solid {ThemeColors.PRIMARY};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {ThemeColors.BORDER_DARK};
        border-radius: 3px;
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ThemeColors.PRIMARY};
        border-color: {ThemeColors.PRIMARY};
    }}
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {ThemeColors.BORDER_DARK};
        border-radius: 9px;
    }}
    QRadioButton::indicator:checked {{
        background-color: {ThemeColors.PRIMARY};
        border-color: {ThemeColors.PRIMARY};
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {ThemeColors.BORDER_DARK};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 16px;
        margin: -6px 0;
        background: {ThemeColors.PRIMARY};
        border-radius: 8px;
    }}
    QSplitter::handle {{
        background-color: {ThemeColors.BORDER_DARK};
    }}
    QToolTip {{
        background-color: {ThemeColors.BACKGROUND_DARK_SECONDARY};
        color: {ThemeColors.TEXT_DARK};
        border: 1px solid {ThemeColors.BORDER_DARK};
        padding: 4px 8px;
    }}
    """


class ThemeManager:
    _instance: ThemeManager | None = None
    _current_mode: ThemeMode = ThemeMode.DARK

    def __new__(cls) -> ThemeManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._settings_path = Path.home() / ".attention_studio" / "theme.json"
            self._load_settings()

    def _load_settings(self) -> None:
        if self._settings_path.exists():
            try:
                with open(self._settings_path) as f:
                    data = json.load(f)
                    mode = data.get("mode", "dark")
                    self._current_mode = ThemeMode(mode)
            except (json.JSONDecodeError, ValueError):
                self._current_mode = ThemeMode.DARK

    def _save_settings(self) -> None:
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._settings_path, "w") as f:
            json.dump({"mode": self._current_mode.value}, f)

    @property
    def mode(self) -> ThemeMode:
        return self._current_mode

    @property
    def is_dark(self) -> bool:
        return self._current_mode == ThemeMode.DARK

    def set_mode(self, mode: ThemeMode) -> None:
        self._current_mode = mode
        self._save_settings()

    def toggle(self) -> None:
        if self._current_mode == ThemeMode.DARK:
            self.set_mode(ThemeMode.LIGHT)
        else:
            self.set_mode(ThemeMode.DARK)

    @property
    def stylesheet(self) -> str:
        if self._current_mode == ThemeMode.LIGHT:
            return _get_light_stylesheet()
        return _get_dark_stylesheet()

    def get_color(self, color_name: str) -> str:
        colors = {
            "primary": ThemeColors.PRIMARY,
            "primary_hover": ThemeColors.PRIMARY_HOVER,
            "primary_active": ThemeColors.PRIMARY_ACTIVE,
            "secondary": ThemeColors.SECONDARY,
            "accent": ThemeColors.ACCENT,
            "success": ThemeColors.SUCCESS,
            "warning": ThemeColors.WARNING,
            "error": ThemeColors.ERROR,
            "info": ThemeColors.INFO,
        }
        return colors.get(color_name, ThemeColors.PRIMARY)

    def apply_to_app(self, app: Any) -> None:
        if hasattr(app, "setStyleSheet"):
            app.setStyleSheet(self.stylesheet)
