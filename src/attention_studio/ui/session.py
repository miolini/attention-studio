from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SessionState:
    name: str
    path: str
    created_at: str
    modified_at: str
    model_name: str = ""
    dataset_name: str = ""
    graph_nodes: int = 0
    graph_edges: int = 0
    viewport_position: dict[str, float] = field(default_factory=dict)
    selected_nodes: list[str] = field(default_factory=list)
    open_tabs: list[str] = field(default_factory=list)
    settings_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "viewport_position": self.viewport_position,
            "selected_nodes": self.selected_nodes,
            "open_tabs": self.open_tabs,
            "settings_overrides": self.settings_overrides,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            name=data.get("name", "Untitled"),
            path=data.get("path", ""),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
            model_name=data.get("model_name", ""),
            dataset_name=data.get("dataset_name", ""),
            graph_nodes=data.get("graph_nodes", 0),
            graph_edges=data.get("graph_edges", 0),
            viewport_position=data.get("viewport_position", {}),
            selected_nodes=data.get("selected_nodes", []),
            open_tabs=data.get("open_tabs", []),
            settings_overrides=data.get("settings_overrides", {}),
        )


class SessionManager:
    _instance: SessionManager | None = None
    _current_session: SessionState | None = None
    _is_dirty: bool = False
    _workspace_dir: Path | None = None

    def __new__(cls) -> SessionManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._workspace_dir = Path.home() / ".attention_studio" / "sessions"
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def create_new(self, name: str = "Untitled") -> SessionState:
        timestamp = datetime.now().isoformat()
        session = SessionState(
            name=name,
            path="",
            created_at=timestamp,
            modified_at=timestamp,
        )
        self._current_session = session
        self._is_dirty = True
        return session

    def load(self, path: str | Path) -> SessionState:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        data = json.loads(path.read_text())
        session = SessionState.from_dict(data)
        session.path = str(path)
        self._current_session = session
        self._is_dirty = False
        return session

    def save(self, path: str | Path | None = None) -> SessionState:
        if self._current_session is None:
            raise RuntimeError("No active session to save")

        if path:
            self._current_session.path = str(path)
        elif not self._current_session.path:
            raise ValueError("No save path specified")

        save_path = Path(self._current_session.path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self._current_session.modified_at = datetime.now().isoformat()
        save_path.write_text(json.dumps(self._current_session.to_dict(), indent=2))
        self._is_dirty = False
        return self._current_session

    def save_auto(self) -> SessionState | None:
        if self._current_session is None:
            return None

        auto_save_dir = self._workspace_dir / "autosave"
        auto_save_dir.mkdir(parents=True, exist_ok=True)

        if not self._current_session.name:
            self._current_session.name = "Untitled"

        auto_path = auto_save_dir / f"{self._current_session.name}.session.json"
        return self.save(auto_path)

    @property
    def current_session(self) -> SessionState | None:
        return self._current_session

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    def mark_dirty(self) -> None:
        self._is_dirty = True

    def update_model_info(self, model_name: str) -> None:
        if self._current_session:
            self._current_session.model_name = model_name
            self.mark_dirty()

    def update_dataset_info(self, dataset_name: str) -> None:
        if self._current_session:
            self._current_session.dataset_name = dataset_name
            self.mark_dirty()

    def update_graph_info(self, nodes: int, edges: int) -> None:
        if self._current_session:
            self._current_session.graph_nodes = nodes
            self._current_session.graph_edges = edges
            self.mark_dirty()

    def update_viewport(self, position: dict[str, float]) -> None:
        if self._current_session:
            self._current_session.viewport_position = position
            self.mark_dirty()

    def update_selected_nodes(self, nodes: list[str]) -> None:
        if self._current_session:
            self._current_session.selected_nodes = nodes
            self.mark_dirty()

    def update_open_tabs(self, tabs: list[str]) -> None:
        if self._current_session:
            self._current_session.open_tabs = tabs
            self.mark_dirty()

    def list_sessions(self) -> list[SessionState]:
        sessions = []
        for path in self._workspace_dir.glob("*.session.json"):
            try:
                data = json.loads(path.read_text())
                session = SessionState.from_dict(data)
                session.path = str(path)
                sessions.append(session)
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(sessions, key=lambda s: s.modified_at, reverse=True)

    def delete_session(self, path: str | Path) -> bool:
        path = Path(path)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_recent_sessions(self, limit: int = 10) -> list[SessionState]:
        all_sessions = self.list_sessions()
        return all_sessions[:limit]

    def close(self) -> None:
        if self._current_session and self._is_dirty:
            self.save_auto()
        self._current_session = None
        self._is_dirty = False


def get_session_manager() -> SessionManager:
    return SessionManager()
