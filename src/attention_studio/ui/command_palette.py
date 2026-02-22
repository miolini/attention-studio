from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Command:
    id: str
    title: str
    description: str = ""
    shortcut: str = ""
    category: str = "General"
    icon: str = ""


class CommandPalette:
    def __init__(self):
        self.commands: dict[str, Command] = {}
        self._callbacks: dict[str, Callable] = {}

    def register(
        self,
        command_id: str,
        title: str,
        callback: Callable,
        description: str = "",
        shortcut: str = "",
        category: str = "General",
        icon: str = "",
    ) -> None:
        command = Command(
            id=command_id,
            title=title,
            description=description,
            shortcut=shortcut,
            category=category,
            icon=icon,
        )
        self.commands[command_id] = command
        self._callbacks[command_id] = callback

    def unregister(self, command_id: str) -> bool:
        if command_id in self.commands:
            del self.commands[command_id]
            if command_id in self._callbacks:
                del self._callbacks[command_id]
            return True
        return False

    def execute(self, command_id: str) -> bool:
        if command_id in self._callbacks:
            callback = self._callbacks[command_id]
            callback()
            return True
        return False

    def search(self, query: str) -> list[Command]:
        if not query:
            return list(self.commands.values())

        query_lower = query.lower()
        results = []

        for command in self.commands.values():
            if (query_lower in command.title.lower() or
                query_lower in command.description.lower() or
                query_lower in command.category.lower()):
                results.append(command)

        results.sort(key=lambda c: (
            c.title.lower().startswith(query_lower),
            c.category,
            c.title,
        ), reverse=True)

        return results

    def get_by_category(self, category: str) -> list[Command]:
        return [c for c in self.commands.values() if c.category == category]

    def get_all_categories(self) -> list[str]:
        return sorted({c.category for c in self.commands.values()})

    def get_all_commands(self) -> list[Command]:
        return list(self.commands.values())

    def has_command(self, command_id: str) -> bool:
        return command_id in self.commands

    def get_command(self, command_id: str) -> Command | None:
        return self.commands.get(command_id)


class CommandPaletteManager:
    _instance: CommandPalette | None = None

    def __new__(cls) -> CommandPalette:
        if cls._instance is None:
            cls._instance = CommandPalette()
        return cls._instance
