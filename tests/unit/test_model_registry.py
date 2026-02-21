from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import os

from attention_studio.core.model_registry import (
    ModelVersion,
    ModelRegistry,
    ModelRegistryManager,
)


class TestModelVersion:
    def test_version_creation(self):
        version = ModelVersion(
            version_id="v0001",
            model_path="/models/test.pt",
            created_at="2024-01-01T00:00:00",
        )
        assert version.version_id == "v0001"
        assert version.model_path == "/models/test.pt"


class TestModelRegistry:
    def test_registry_creation(self):
        registry = ModelRegistry(name="test-registry")
        assert registry.name == "test-registry"
        assert len(registry.versions) == 0


class TestModelRegistryManager:
    def test_manager_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manager = ModelRegistryManager("test", storage_path=path)
            assert manager.registry_name == "test"

    def test_register_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path = path / "model.pt"
            model_path.write_text("model data")

            manager = ModelRegistryManager("test", storage_path=path)
            version_id = manager.register_model(
                str(model_path), metadata={"accuracy": 0.95}
            )
            assert version_id == "v0001"
            assert manager.registry.current_version == "v0001"

    def test_get_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path = path / "model.pt"
            model_path.write_text("model data")

            manager = ModelRegistryManager("test", storage_path=path)
            version_id = manager.register_model(str(model_path))

            version = manager.get_version(version_id)
            assert version is not None
            assert version.version_id == version_id

    def test_list_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path1 = path / "model1.pt"
            model_path2 = path / "model2.pt"
            model_path1.write_text("model1")
            model_path2.write_text("model2")

            manager = ModelRegistryManager("test", storage_path=path)
            manager.register_model(str(model_path1))
            manager.register_model(str(model_path2))

            versions = manager.list_versions()
            assert len(versions) == 2

    def test_set_current_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path1 = path / "model1.pt"
            model_path2 = path / "model2.pt"
            model_path1.write_text("model1")
            model_path2.write_text("model2")

            manager = ModelRegistryManager("test", storage_path=path)
            v1 = manager.register_model(str(model_path1))
            v2 = manager.register_model(str(model_path2))

            result = manager.set_current_version(v1)
            assert result is True
            assert manager.registry.current_version == v1

    def test_delete_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path1 = path / "model1.pt"
            model_path2 = path / "model2.pt"
            model_path1.write_text("model1")
            model_path2.write_text("model2")

            manager = ModelRegistryManager("test", storage_path=path)
            v1 = manager.register_model(str(model_path1))
            manager.register_model(str(model_path2))

            result = manager.delete_version(v1)
            assert result is True
            assert len(manager.list_versions()) == 1

    def test_get_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path1 = path / "model1.pt"
            model_path2 = path / "model2.pt"
            model_path1.write_text("model1")
            model_path2.write_text("model2")

            manager = ModelRegistryManager("test", storage_path=path)
            v1 = manager.register_model(str(model_path1))
            v2 = manager.register_model(str(model_path2))

            lineage = manager.get_lineage(v2)
            assert len(lineage) == 2
            assert lineage[0].version_id == v2
            assert lineage[1].version_id == v1
