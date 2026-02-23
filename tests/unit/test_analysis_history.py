from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from attention_studio.core.analysis_history import AnalysisHistory, AnalysisResult


class TestAnalysisResult:
    def test_default_values(self):
        result = AnalysisResult(
            id="test_1",
            timestamp="2024-01-01T00:00:00",
            analysis_type="feature_extraction",
            model_name="gpt2",
            prompt="test prompt",
        )
        assert result.id == "test_1"
        assert result.metrics == {}
        assert result.features == []
        assert result.graph_data == {}
        assert result.notes == ""
        assert result.tags == []

    def test_to_dict(self):
        result = AnalysisResult(
            id="test_1",
            timestamp="2024-01-01T00:00:00",
            analysis_type="feature_extraction",
            model_name="gpt2",
            prompt="test prompt",
            metrics={"accuracy": 0.95},
            tags=["important"],
        )
        data = result.to_dict()
        assert data["id"] == "test_1"
        assert data["metrics"]["accuracy"] == 0.95
        assert data["tags"] == ["important"]

    def test_from_dict(self):
        data = {
            "id": "test_1",
            "timestamp": "2024-01-01T00:00:00",
            "analysis_type": "feature_extraction",
            "model_name": "gpt2",
            "prompt": "test prompt",
            "metrics": {"accuracy": 0.95},
            "features": [{"name": "feat1"}],
            "notes": "test notes",
            "tags": ["important"],
        }
        result = AnalysisResult.from_dict(data)
        assert result.id == "test_1"
        assert result.metrics["accuracy"] == 0.95
        assert result.features[0]["name"] == "feat1"
        assert result.notes == "test notes"
        assert result.tags == ["important"]


class TestAnalysisHistory:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture(autouse=True)
    def reset_global(self):
        import attention_studio.core.analysis_history as ah
        ah._history = None
        yield
        ah._history = None

    def test_init(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        assert history._storage_dir == temp_dir
        assert len(history) == 0

    def test_add(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        result = history.add(
            analysis_type="feature_extraction",
            model_name="gpt2",
            prompt="test prompt",
            metrics={"accuracy": 0.95},
        )
        assert result.id.startswith("analysis_")
        assert result.analysis_type == "feature_extraction"
        assert result.metrics["accuracy"] == 0.95
        assert len(history) == 1

    def test_persistence(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(
            analysis_type="feature_extraction",
            model_name="gpt2",
            prompt="test prompt",
        )

        history2 = AnalysisHistory(temp_dir)
        assert len(history2) == 1
        assert history2[0].analysis_type == "feature_extraction"

    def test_get(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        added = history.add(
            analysis_type="feature_extraction",
            model_name="gpt2",
            prompt="test prompt",
        )

        result = history.get(added.id)
        assert result is not None
        assert result.id == added.id

    def test_get_nonexistent(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        result = history.get("nonexistent_id")
        assert result is None

    def test_list_all(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="type1", model_name="gpt2", prompt="p1")
        history.add(analysis_type="type2", model_name="gpt2", prompt="p2")

        results = history.list_all()
        assert len(results) == 2

    def test_list_by_type(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="feature_extraction", model_name="gpt2", prompt="p1")
        history.add(analysis_type="feature_extraction", model_name="gpt2", prompt="p2")
        history.add(analysis_type="ablation", model_name="gpt2", prompt="p3")

        results = history.list_by_type("feature_extraction")
        assert len(results) == 2

    def test_list_by_model(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p1")
        history.add(analysis_type="extraction", model_name="gpt-neo", prompt="p2")

        results = history.list_by_model("gpt2")
        assert len(results) == 1

    def test_list_by_tag(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p1", tags=["important"])
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p2", tags=["review"])

        results = history.list_by_tag("important")
        assert len(results) == 1

    def test_search(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="The cat sat on the mat")
        history.add(analysis_type="ablation", model_name="gpt2", prompt="Hello world")

        results = history.search("cat")
        assert len(results) == 1
        assert "cat" in results[0].prompt

    def test_delete(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        added = history.add(analysis_type="extraction", model_name="gpt2", prompt="p1")

        result = history.delete(added.id)
        assert result is True
        assert len(history) == 0

    def test_delete_nonexistent(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        result = history.delete("nonexistent_id")
        assert result is False

    def test_clear(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p1")
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p2")

        history.clear()
        assert len(history) == 0

    def test_export_json(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p1", metrics={"acc": 0.9})

        export_path = temp_dir / "export.json"
        history.export_json(export_path)

        data = json.loads(export_path.read_text())
        assert data["count"] == 1
        assert data["results"][0]["metrics"]["acc"] == 0.9

    def test_export_csv(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(
            analysis_type="extraction",
            model_name="gpt2",
            prompt="test prompt",
            metrics={"accuracy": 0.95, "loss": 0.05},
        )

        export_path = temp_dir / "export.csv"
        history.export_csv(export_path)

        content = export_path.read_text()
        assert "analysis_type" in content
        assert "metric_accuracy" in content

    def test_get_summary(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p1")
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p2")
        history.add(analysis_type="ablation", model_name="gpt-neo", prompt="p3")

        summary = history.get_summary()
        assert summary["total"] == 3
        assert summary["by_type"]["extraction"] == 2
        assert summary["by_type"]["ablation"] == 1

    def test_iteration(self, temp_dir):
        history = AnalysisHistory(temp_dir)
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p1")
        history.add(analysis_type="extraction", model_name="gpt2", prompt="p2")

        count = 0
        for result in history:
            count += 1
        assert count == 2
