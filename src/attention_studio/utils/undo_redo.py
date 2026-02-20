from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    name: str
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]
    data: dict[str, Any] = field(default_factory=dict)


class UndoRedoManager:
    def __init__(self, max_history: int = 50):
        self._undo_stack: deque[Action] = deque(maxlen=max_history)
        self._redo_stack: deque[Action] = deque(maxlen=max_history)
        self._max_history = max_history
        self._paused = False
        self._actionListeners: list[Callable[[str], None]] = []

    def add_action(self, action: Action):
        if self._paused:
            return

        self._undo_stack.append(action)
        self._redo_stack.clear()

        for listener in self._actionListeners:
            listener("add")

    def undo(self) -> bool:
        if not self._undo_stack:
            return False

        action = self._undo_stack.pop()
        try:
            action.undo_fn()
            self._redo_stack.append(action)

            for listener in self._actionListeners:
                listener("undo")

            return True
        except Exception:
            return False

    def redo(self) -> bool:
        if not self._redo_stack:
            return False

        action = self._redo_stack.pop()
        try:
            action.redo_fn()
            self._undo_stack.append(action)

            for listener in self._actionListeners:
                listener("redo")

            return True
        except Exception:
            return False

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def get_undo_description(self) -> str:
        if self._undo_stack:
            return self._undo_stack[-1].name
        return ""

    def get_redo_description(self) -> str:
        if self._redo_stack:
            return self._redo_stack[-1].name
        return ""

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

        for listener in self._actionListeners:
            listener("clear")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def add_listener(self, listener: Callable[[str], None]):
        self._actionListeners.append(listener)

    def remove_listener(self, listener: Callable[[str], None]):
        if listener in self._actionListeners:
            self._actionListeners.remove(listener)


class GraphStateManager:
    def __init__(self, max_history: int = 30):
        self._undo_redo = UndoRedoManager(max_history)

    @property
    def can_undo(self) -> bool:
        return self._undo_redo.can_undo()

    @property
    def can_redo(self) -> bool:
        return self._undo_redo.can_redo()

    def record_node_moved(
        self,
        node_id: str,
        old_pos: tuple[float, float],
        new_pos: tuple[float, float],
    ):
        def undo():
            pass

        def redo():
            pass

        action = Action(
            name=f"Move node {node_id}",
            undo_fn=undo,
            redo_fn=redo,
            data={"node_id": node_id, "old_pos": old_pos, "new_pos": new_pos},
        )
        self._undo_redo.add_action(action)

    def record_edge_added(self, edge: tuple[str, str], weight: float):
        action = Action(
            name=f"Add edge {edge[0]} -> {edge[1]}",
            undo_fn=lambda: None,
            redo_fn=lambda: None,
            data={"edge": edge, "weight": weight},
        )
        self._undo_redo.add_action(action)

    def record_edge_removed(self, edge: tuple[str, str], weight: float):
        action = Action(
            name=f"Remove edge {edge[0]} -> {edge[1]}",
            undo_fn=lambda: None,
            redo_fn=lambda: None,
            data={"edge": edge, "weight": weight},
        )
        self._undo_redo.add_action(action)

    def record_filter_applied(self, old_filters: dict, new_filters: dict):
        action = Action(
            name="Apply filters",
            undo_fn=lambda: None,
            redo_fn=lambda: None,
            data={"old_filters": old_filters, "new_filters": new_filters},
        )
        self._undo_redo.add_action(action)

    def record_layout_changed(self, old_layout: str, new_layout: str):
        action = Action(
            name=f"Change layout: {old_layout} -> {new_layout}",
            undo_fn=lambda: None,
            redo_fn=lambda: None,
            data={"old_layout": old_layout, "new_layout": new_layout},
        )
        self._undo_redo.add_action(action)

    def undo(self) -> bool:
        return self._undo_redo.undo()

    def redo(self) -> bool:
        return self._undo_redo.redo()

    def get_undo_description(self) -> str:
        return self._undo_redo.get_undo_description()

    def get_redo_description(self) -> str:
        return self._undo_redo.get_redo_description()
