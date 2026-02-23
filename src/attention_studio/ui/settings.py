from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings


class GeneralSettings(BaseModel):
    auto_save: bool = True
    auto_save_interval: int = 300
    show_welcome: bool = True
    recent_sessions_limit: int = 10
    confirm_quit: bool = True


class ModelSettings(BaseModel):
    default_model: str = "gpt2"
    device: str = "auto"
    dtype: str = "float16"
    cache_dir: str = "~/.cache/huggingface"
    max_memory_gb: int = 8


class VisualizationSettings(BaseModel):
    theme: str = "dark"
    animation_enabled: bool = True
    node_size: int = 12
    edge_transparency: float = 0.6
    show_tooltips: bool = True
    layout_algorithm: str = "force_directed"


class NetworkSettings(BaseModel):
    enable_telemetry: bool = False
    check_updates: bool = True


class AppSettings(BaseSettings):
    model_config = ConfigDict(settings_file="settings.json")

    general: GeneralSettings = GeneralSettings()
    model: ModelSettings = ModelSettings()
    visualization: VisualizationSettings = VisualizationSettings()
    network: NetworkSettings = NetworkSettings()

    def to_dict(self) -> dict[str, Any]:
        return {
            "general": self.general.model_dump(),
            "model": self.model.model_dump(),
            "visualization": self.visualization.model_dump(),
            "network": self.network.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppSettings:
        return cls(
            general=GeneralSettings(**data.get("general", {})),
            model=ModelSettings(**data.get("model", {})),
            visualization=VisualizationSettings(**data.get("visualization", {})),
            network=NetworkSettings(**data.get("network", {})),
        )


class SettingsManager:
    _instance: SettingsManager | None = None
    _settings: AppSettings | None = None

    def __new__(cls) -> SettingsManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._path: Path | None = None
            self._initialized = True

    def initialize(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            config_dir = Path.home() / ".attention_studio"
        config_dir.mkdir(parents=True, exist_ok=True)
        self._path = config_dir / "settings.json"
        self.load()

    def load(self) -> AppSettings:
        if self._path is None:
            self.initialize()

        if self._path and self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._settings = AppSettings.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                self._settings = AppSettings()
        else:
            self._settings = AppSettings()
        return self._settings

    def save(self) -> None:
        if self._path is None or self._settings is None:
            return
        self._path.write_text(json.dumps(self._settings.to_dict(), indent=2))

    @property
    def settings(self) -> AppSettings:
        if self._settings is None:
            self.load()
        return self._settings

    def update_general(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._settings.general, key):
                setattr(self._settings.general, key, value)

    def update_model(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._settings.model, key):
                setattr(self._settings.model, key, value)

    def update_visualization(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._settings.visualization, key):
                setattr(self._settings.visualization, key, value)

    def update_network(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._settings.network, key):
                setattr(self._settings.network, key, value)

    def reset(self) -> None:
        self._settings = AppSettings()
        self.save()

    def get_recent_sessions(self) -> list[str]:
        general = self._settings.general if self._settings else GeneralSettings()
        sessions_file = self._path.parent / "recent_sessions.json"
        if sessions_file.exists():
            try:
                sessions = json.loads(sessions_file.read_text())
                return sessions[: general.recent_sessions_limit]
            except json.JSONDecodeError:
                return []
        return []

    def add_recent_session(self, session_path: str) -> None:
        general = self._settings.general if self._settings else GeneralSettings()
        sessions = self.get_recent_sessions()
        if session_path in sessions:
            sessions.remove(session_path)
        sessions.insert(0, session_path)
        sessions = sessions[: general.recent_sessions_limit]
        sessions_file = self._path.parent / "recent_sessions.json"
        sessions_file.write_text(json.dumps(sessions))


def get_settings_manager() -> SettingsManager:
    manager = SettingsManager()
    if manager._path is None:
        manager.initialize()
    return manager


def get_settings() -> AppSettings:
    return get_settings_manager().settings
