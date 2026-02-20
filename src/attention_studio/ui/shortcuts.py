from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class KeyboardShortcut:
    key: str
    modifiers: list[str]
    action: str
    description: str
    callback: Callable | None = None


class KeyboardShortcutManager:
    DEFAULT_SHORTCUTS = [
        ("L", ["Ctrl"], "Load Model", "load_model"),
        ("B", ["Ctrl"], "Build CRM", "build_crm"),
        ("T", ["Ctrl"], "Train", "train"),
        ("E", ["Ctrl"], "Extract Features", "extract_features"),
        ("G", ["Ctrl"], "Build Graph", "build_graph"),
        ("F", ["Ctrl"], "Find Circuits", "find_circuits"),
        ("S", ["Ctrl"], "Save Session", "save_session"),
        ("O", ["Ctrl"], "Open Session", "open_session"),
        ("N", ["Ctrl"], "New Analysis", "new_analysis"),
        ("Q", ["Ctrl"], "Quit", "quit"),
        ("1", ["Ctrl"], "Tab: Model", "tab_model"),
        ("2", ["Ctrl"], "Tab: Graph", "tab_graph"),
        ("3", ["Ctrl"], "Tab: Features", "tab_features"),
        ("4", ["Ctrl"], "Tab: Compare", "tab_compare"),
        ("5", ["Ctrl"], "Tab: Agent", "tab_agent"),
        ("/", ["Ctrl"], "Focus Search", "focus_search"),
        ("H", ["Ctrl"], "Toggle Sidebar", "toggle_sidebar"),
        ("P", ["Ctrl"], "Toggle Console", "toggle_console"),
    ]

    def __init__(self):
        self._shortcuts: dict[str, KeyboardShortcut] = {}
        self._action_handlers: dict[str, Callable] = {}
        self._setup_default_shortcuts()

    def _setup_default_shortcuts(self):
        for key, modifiers, description, action in self.DEFAULT_SHORTCUTS:
            self._shortcuts[f"{'+'.join(modifiers)}+{key}"] = KeyboardShortcut(
                key=key,
                modifiers=modifiers,
                action=action,
                description=description,
            )

    def register_handler(self, action: str, handler: Callable):
        self._action_handlers[action] = handler

    def trigger(self, action: str):
        if action in self._action_handlers:
            handler = self._action_handlers[action]
            if callable(handler):
                handler()

    def get_shortcut_for_action(self, action: str) -> KeyboardShortcut | None:
        for shortcut in self._shortcuts.values():
            if shortcut.action == action:
                return shortcut
        return None

    def get_all_shortcuts(self) -> list[KeyboardShortcut]:
        return list(self._shortcuts.values())

    def get_shortcut_text(self, action: str) -> str:
        shortcut = self.get_shortcut_for_action(action)
        if shortcut:
            return f"{'+'.join(shortcut.modifiers)}+{shortcut.key}"
        return ""


_shortcut_manager: KeyboardShortcutManager | None = None


def get_shortcut_manager() -> KeyboardShortcutManager:
    global _shortcut_manager
    if _shortcut_manager is None:
        _shortcut_manager = KeyboardShortcutManager()
    return _shortcut_manager
