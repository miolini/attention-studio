from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from attention_studio.ui.settings import (
    AppSettings,
    GeneralSettings,
    ModelSettings,
    SettingsManager,
    VisualizationSettings,
    NetworkSettings,
)


class TestGeneralSettings:
    def test_defaults(self):
        settings = GeneralSettings()
        assert settings.auto_save is True
        assert settings.auto_save_interval == 300
        assert settings.show_welcome is True
        assert settings.recent_sessions_limit == 10
        assert settings.confirm_quit is True

    def test_custom_values(self):
        settings = GeneralSettings(
            auto_save=False,
            auto_save_interval=600,
            show_welcome=False,
        )
        assert settings.auto_save is False
        assert settings.auto_save_interval == 600
        assert settings.show_welcome is False


class TestModelSettings:
    def test_defaults(self):
        settings = ModelSettings()
        assert settings.default_model == "gpt2"
        assert settings.device == "auto"
        assert settings.dtype == "float16"
        assert settings.cache_dir == "~/.cache/huggingface"
        assert settings.max_memory_gb == 8

    def test_custom_values(self):
        settings = ModelSettings(
            default_model="gpt2-large",
            device="cuda",
            dtype="float32",
        )
        assert settings.default_model == "gpt2-large"
        assert settings.device == "cuda"
        assert settings.dtype == "float32"


class TestVisualizationSettings:
    def test_defaults(self):
        settings = VisualizationSettings()
        assert settings.theme == "dark"
        assert settings.animation_enabled is True
        assert settings.node_size == 12
        assert settings.edge_transparency == 0.6
        assert settings.show_tooltips is True
        assert settings.layout_algorithm == "force_directed"

    def test_custom_values(self):
        settings = VisualizationSettings(
            theme="light",
            animation_enabled=False,
            node_size=20,
        )
        assert settings.theme == "light"
        assert settings.animation_enabled is False
        assert settings.node_size == 20


class TestNetworkSettings:
    def test_defaults(self):
        settings = NetworkSettings()
        assert settings.enable_telemetry is False
        assert settings.check_updates is True


class TestAppSettings:
    def test_defaults(self):
        settings = AppSettings()
        assert isinstance(settings.general, GeneralSettings)
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.visualization, VisualizationSettings)
        assert isinstance(settings.network, NetworkSettings)

    def test_to_dict(self):
        settings = AppSettings()
        data = settings.to_dict()
        assert "general" in data
        assert "model" in data
        assert "visualization" in data
        assert "network" in data

    def test_from_dict(self):
        data = {
            "general": {"auto_save": False},
            "model": {"default_model": "test-model"},
            "visualization": {"theme": "light"},
            "network": {"enable_telemetry": True},
        }
        settings = AppSettings.from_dict(data)
        assert settings.general.auto_save is False
        assert settings.model.default_model == "test-model"
        assert settings.visualization.theme == "light"
        assert settings.network.enable_telemetry is True


class TestSettingsManager:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_singleton(self):
        manager1 = SettingsManager()
        manager2 = SettingsManager()
        assert manager1 is manager2

    def test_initialize(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        assert manager._path == temp_dir / "settings.json"

    def test_save_and_load(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.settings.general.auto_save = False
        manager.settings.model.default_model = "custom-model"
        manager.save()

        manager2 = SettingsManager()
        manager2.initialize(temp_dir)
        manager2.load()
        assert manager2.settings.general.auto_save is False
        assert manager2.settings.model.default_model == "custom-model"

    def test_update_general(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.update_general(auto_save=False, auto_save_interval=600)
        assert manager.settings.general.auto_save is False
        assert manager.settings.general.auto_save_interval == 600

    def test_update_model(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.update_model(default_model="gpt-neo", device="cpu")
        assert manager.settings.model.default_model == "gpt-neo"
        assert manager.settings.model.device == "cpu"

    def test_update_visualization(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.update_visualization(theme="light", node_size=24)
        assert manager.settings.visualization.theme == "light"
        assert manager.settings.visualization.node_size == 24

    def test_reset(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.settings.general.auto_save = False
        manager.reset()
        assert manager.settings.general.auto_save is True

    def test_recent_sessions(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.add_recent_session("/path/to/session1.json")
        manager.add_recent_session("/path/to/session2.json")
        sessions = manager.get_recent_sessions()
        assert "/path/to/session1.json" in sessions
        assert "/path/to/session2.json" in sessions

    def test_recent_sessions_limit(self, temp_dir):
        manager = SettingsManager()
        manager.initialize(temp_dir)
        manager.update_general(recent_sessions_limit=3)
        for i in range(5):
            manager.add_recent_session(f"/path/to/session{i}.json")
        sessions = manager.get_recent_sessions()
        assert len(sessions) == 3
