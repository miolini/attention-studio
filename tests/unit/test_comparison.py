from __future__ import annotations

import pytest

from attention_studio.core.comparison import (
    ComparisonEngine,
    ComparisonItem,
    compare_models,
    compare_prompts,
    compare_datasets,
)


class TestComparisonItem:
    def test_default_values(self):
        item = ComparisonItem(name="test")
        assert item.name == "test"
        assert item.metrics == {}
        assert item.metadata == {}

    def test_with_values(self):
        item = ComparisonItem(
            name="model1",
            metrics={"accuracy": 0.95},
            metadata={"type": "model"},
        )
        assert item.metrics["accuracy"] == 0.95
        assert item.metadata["type"] == "model"


class TestComparisonEngine:
    def test_add_item(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})

        assert len(engine) == 1
        assert engine[0].name == "model1"

    def test_add_items(self):
        engine = ComparisonEngine()
        items = [
            ComparisonItem(name="model1", metrics={"acc": 0.9}),
            ComparisonItem(name="model2", metrics={"acc": 0.8}),
        ]
        engine.add_items(items)

        assert len(engine) == 2

    def test_clear(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})
        engine.clear()

        assert len(engine) == 0

    def test_get_item(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})

        item = engine.get_item("model1")
        assert item is not None
        assert item.name == "model1"

    def test_get_item_nonexistent(self):
        engine = ComparisonEngine()
        item = engine.get_item("nonexistent")
        assert item is None

    def test_list_items(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})
        engine.add_item("model2", {"accuracy": 0.85})

        items = engine.list_items()
        assert len(items) == 2

    def test_get_metric_names(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95, "loss": 0.05})
        engine.add_item("model2", {"accuracy": 0.85, "f1": 0.9})

        names = engine.get_metric_names()
        assert "accuracy" in names
        assert "loss" in names
        assert "f1" in names

    def test_compare_metric(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})
        engine.add_item("model2", {"accuracy": 0.85})

        result = engine.compare_metric("accuracy")
        assert result is not None
        assert result.metric_name == "accuracy"
        assert len(result.values) == 2

    def test_compare_metric_nonexistent(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})

        result = engine.compare_metric("nonexistent")
        assert result is None

    def test_compare_all_metrics(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95, "loss": 0.05})
        engine.add_item("model2", {"accuracy": 0.85, "loss": 0.15})

        results = engine.compare_all_metrics()
        assert len(results) == 2

    def test_rank_items_ascending(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})
        engine.add_item("model2", {"accuracy": 0.85})
        engine.add_item("model3", {"accuracy": 0.90})

        ranked = engine.rank_items("accuracy", ascending=False)
        assert ranked[0] == ("model1", 0.95)
        assert ranked[1] == ("model3", 0.90)
        assert ranked[2] == ("model2", 0.85)

    def test_rank_items_descending(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})
        engine.add_item("model2", {"accuracy": 0.85})

        ranked = engine.rank_items("accuracy", ascending=True)
        assert ranked[0] == ("model2", 0.85)

    def test_get_best_item_higher_better(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95})
        engine.add_item("model2", {"accuracy": 0.85})

        best = engine.get_best_item("accuracy", higher_is_better=True)
        assert best == ("model1", 0.95)

    def test_get_best_item_lower_better(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"loss": 0.05})
        engine.add_item("model2", {"loss": 0.15})

        best = engine.get_best_item("loss", higher_is_better=False)
        assert best == ("model1", 0.05)

    def test_get_summary(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95, "loss": 0.05})
        engine.add_item("model2", {"accuracy": 0.85, "loss": 0.15})

        summary = engine.get_summary()
        assert summary["count"] == 2
        assert "accuracy" in summary["metrics"]

    def test_correlation(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95, "loss": 0.05})
        engine.add_item("model2", {"accuracy": 0.85, "loss": 0.15})
        engine.add_item("model3", {"accuracy": 0.75, "loss": 0.25})

        corr = engine.compute_correlation("accuracy", "loss")
        assert corr is not None
        assert corr < 0

    def test_correlation_single_item(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95, "loss": 0.05})

        corr = engine.compute_correlation("accuracy", "loss")
        assert corr is None

    def test_find_outliers(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"value": 0.0})
        engine.add_item("model2", {"value": 0.1})
        engine.add_item("model3", {"value": 0.2})
        engine.add_item("model4", {"value": 10.0})

        outliers = engine.find_outliers("value", threshold=0.5)
        assert len(outliers) >= 1
        names = [o[0] for o in outliers]
        assert "model4" in names

    def test_to_dict(self):
        engine = ComparisonEngine()
        engine.add_item("model1", {"accuracy": 0.95}, {"type": "model"})

        data = engine.to_dict()
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "model1"

    def test_from_dict(self):
        data = {
            "items": [
                {"name": "model1", "metrics": {"accuracy": 0.95}},
                {"name": "model2", "metrics": {"accuracy": 0.85}},
            ]
        }
        engine = ComparisonEngine.from_dict(data)

        assert len(engine) == 2
        assert engine[0].name == "model1"


class TestHelperFunctions:
    def test_compare_models(self):
        model_results = {
            "gpt2": {"accuracy": 0.95, "loss": 0.05},
            "gpt-neo": {"accuracy": 0.90, "loss": 0.10},
        }
        engine = compare_models(model_results)

        assert len(engine) == 2
        assert engine.get_item("gpt2") is not None

    def test_compare_models_filtered(self):
        model_results = {
            "gpt2": {"accuracy": 0.95, "loss": 0.05, "f1": 0.9},
            "gpt-neo": {"accuracy": 0.90, "loss": 0.10, "f1": 0.85},
        }
        engine = compare_models(model_results, metrics_to_compare=["accuracy"])

        assert len(engine) == 2
        metrics = engine.get_item("gpt2").metrics
        assert "accuracy" in metrics

    def test_compare_prompts(self):
        prompt_results = {
            "prompt1": {"accuracy": 0.95},
            "prompt2": {"accuracy": 0.85},
        }
        engine = compare_prompts(prompt_results)

        assert len(engine) == 2
        assert engine[0].metadata["type"] == "prompt"

    def test_compare_datasets(self):
        dataset_results = {
            "dataset1": {"accuracy": 0.95},
            "dataset2": {"accuracy": 0.85},
        }
        engine = compare_datasets(dataset_results)

        assert len(engine) == 2
        assert engine[0].metadata["type"] == "dataset"
