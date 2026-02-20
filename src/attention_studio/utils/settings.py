from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AppSettings:
    theme: str = "dark"
    language: str = "en"
    auto_save_enabled: bool = True
    auto_save_interval: int = 300
    default_model: str = "gpt2"
    default_dataset: str = "c4"
    max_recent_files: int = 10
    window_geometry: dict[str, int] = field(default_factory=dict)
    sidebar_visible: bool = True
    console_visible: bool = True
    last_directory: str = ""


@dataclass
class RecentFile:
    path: str
    name: str
    timestamp: float
    model_name: str = ""


class SettingsManager:
    SETTINGS_FILE = "settings.json"
    RECENT_FILES_FILE = "recent_files.json"
    SESSION_FILE = "session.json"

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.home() / ".config" / "attention_studio"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._settings: AppSettings | None = None
        self._recent_files: list[RecentFile] = []

    @property
    def settings(self) -> AppSettings:
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings

    @property
    def recent_files(self) -> list[RecentFile]:
        return self._recent_files

    def _load_settings(self) -> AppSettings:
        settings_path = self.config_dir / self.SETTINGS_FILE
        if settings_path.exists():
            try:
                with open(settings_path, encoding="utf-8") as f:
                    data = json.load(f)
                return AppSettings(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        return AppSettings()

    def _save_settings(self):
        settings_path = self.config_dir / self.SETTINGS_FILE
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._settings), f, indent=2)

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self._save_settings()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self._settings, key, default)

    def set(self, key: str, value: Any):
        if hasattr(self._settings, key):
            setattr(self._settings, key, value)
            self._save_settings()

    def load_recent_files(self) -> list[RecentFile]:
        recent_path = self.config_dir / self.RECENT_FILES_FILE
        if recent_path.exists():
            try:
                with open(recent_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._recent_files = [RecentFile(**r) for r in data]
            except (json.JSONDecodeError, TypeError):
                self._recent_files = []
        return self._recent_files

    def add_recent_file(self, file_path: Path, model_name: str = ""):
        self.load_recent_files()

        recent = RecentFile(
            path=str(file_path),
            name=file_path.name,
            timestamp=datetime.now().timestamp(),
            model_name=model_name,
        )

        self._recent_files = [r for r in self._recent_files if r.path != str(file_path)]
        self._recent_files.insert(0, recent)

        max_files = self._settings.max_recent_files
        self._recent_files = self._recent_files[:max_files]

        self._save_recent_files()

    def _save_recent_files(self):
        recent_path = self.config_dir / self.RECENT_FILES_FILE
        with open(recent_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._recent_files], f, indent=2)

    def clear_recent_files(self):
        self._recent_files = []
        self._save_recent_files()

    def save_session(self, session_data: dict[str, Any]):
        session_path = self.config_dir / self.SESSION_FILE
        session_data["saved_at"] = datetime.now().isoformat()
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self) -> dict[str, Any] | None:
        session_path = self.config_dir / self.SESSION_FILE
        if session_path.exists():
            try:
                with open(session_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def clear_session(self):
        session_path = self.config_dir / self.SESSION_FILE
        if session_path.exists():
            session_path.unlink()


_settings_manager: SettingsManager | None = None


def get_settings_manager() -> SettingsManager:
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
