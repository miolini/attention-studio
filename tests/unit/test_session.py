from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from attention_studio.ui.session import SessionManager, SessionState


class TestSessionState:
    def test_default_values(self):
        state = SessionState(name="test", path="/test/path", created_at="2024-01-01", modified_at="2024-01-01")
        assert state.name == "test"
        assert state.path == "/test/path"
        assert state.model_name == ""
        assert state.dataset_name == ""
        assert state.graph_nodes == 0
        assert state.graph_edges == 0
        assert state.viewport_position == {}
        assert state.selected_nodes == []
        assert state.open_tabs == []
        assert state.settings_overrides == {}

    def test_to_dict(self):
        state = SessionState(
            name="test",
            path="/test/path",
            created_at="2024-01-01",
            modified_at="2024-01-01",
            model_name="gpt2",
            graph_nodes=10,
        )
        data = state.to_dict()
        assert data["name"] == "test"
        assert data["model_name"] == "gpt2"
        assert data["graph_nodes"] == 10

    def test_from_dict(self):
        data = {
            "name": "test",
            "path": "/test/path",
            "created_at": "2024-01-01",
            "modified_at": "2024-01-01",
            "model_name": "gpt2",
            "dataset_name": "test-data",
            "graph_nodes": 5,
            "graph_edges": 10,
        }
        state = SessionState.from_dict(data)
        assert state.name == "test"
        assert state.model_name == "gpt2"
        assert state.dataset_name == "test-data"
        assert state.graph_nodes == 5
        assert state.graph_edges == 10


class TestSessionManager:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        SessionManager._instance = None
        yield
        SessionManager._instance = None
        SessionManager._current_session = None
        SessionManager._is_dirty = False

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_singleton(self):
        manager1 = SessionManager()
        manager2 = SessionManager()
        assert manager1 is manager2

    def test_create_new(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        session = manager.create_new("Test Session")
        assert session.name == "Test Session"
        assert session.created_at == session.modified_at
        assert manager.current_session is session
        assert manager.is_dirty is True

    def test_save_and_load(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        session = manager.create_new("Test Session")
        session.model_name = "gpt2"
        session.graph_nodes = 100
        session.graph_edges = 50

        save_path = temp_dir / "test.session.json"
        manager.save(save_path)

        SessionManager._instance = None
        manager2 = SessionManager()
        manager2._workspace_dir = temp_dir / "sessions"

        loaded = manager2.load(save_path)

        assert loaded.name == "Test Session"
        assert loaded.model_name == "gpt2"
        assert loaded.graph_nodes == 100
        assert loaded.graph_edges == 50
        assert manager2.is_dirty is False

    def test_load_nonexistent(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"

        with pytest.raises(FileNotFoundError):
            manager.load("/nonexistent/path.json")

    def test_save_without_session(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"

        with pytest.raises(RuntimeError):
            manager.save()

    def test_save_without_path(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager.create_new("Test")

        with pytest.raises(ValueError):
            manager.save()

    def test_update_model_info(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        manager.update_model_info("gpt2-large")
        assert manager.current_session.model_name == "gpt2-large"
        assert manager.is_dirty is True

    def test_update_dataset_info(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        manager.update_dataset_info("custom-data")
        assert manager.current_session.dataset_name == "custom-data"

    def test_update_graph_info(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        manager.update_graph_info(50, 100)
        assert manager.current_session.graph_nodes == 50
        assert manager.current_session.graph_edges == 100

    def test_update_viewport(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        pos = {"x": 100.5, "y": 200.3, "zoom": 1.5}
        manager.update_viewport(pos)
        assert manager.current_session.viewport_position == pos

    def test_update_selected_nodes(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        nodes = ["node1", "node2", "node3"]
        manager.update_selected_nodes(nodes)
        assert manager.current_session.selected_nodes == nodes

    def test_update_open_tabs(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        tabs = ["model", "graph", "features"]
        manager.update_open_tabs(tabs)
        assert manager.current_session.open_tabs == tabs

    def test_mark_dirty(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        assert manager.is_dirty is True
        manager._is_dirty = False
        manager.mark_dirty()
        assert manager.is_dirty is True

    def test_list_sessions(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        session1 = manager.create_new("Session 1")
        session1.model_name = "gpt2"
        manager.save(manager._workspace_dir / "session1.session.json")

        manager.create_new("Session 2")
        manager.update_model_info("gpt-neo")
        manager.save(manager._workspace_dir / "session2.session.json")

        sessions = manager.list_sessions()
        assert len(sessions) >= 2

    def test_delete_session(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        session = manager.create_new("To Delete")
        path = temp_dir / "delete_me.session.json"
        manager.save(path)
        assert path.exists()

        result = manager.delete_session(path)
        assert result is True
        assert not path.exists()

    def test_delete_nonexistent(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"

        result = manager.delete_session("/nonexistent.json")
        assert result is False

    def test_get_recent_sessions(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            manager.create_new(f"Session {i}")
            manager.save(manager._workspace_dir / f"session{i}.session.json")

        recent = manager.get_recent_sessions(limit=3)
        assert len(recent) == 3

    def test_close(self, temp_dir):
        manager = SessionManager()
        manager._workspace_dir = temp_dir / "sessions"
        manager._workspace_dir.mkdir(parents=True, exist_ok=True)

        manager.create_new("Test")
        manager._is_dirty = True
        manager.close()
        assert manager.current_session is None
        assert manager.is_dirty is False
