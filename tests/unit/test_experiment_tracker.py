import pytest
import tempfile
from pathlib import Path
from attention_studio.utils.experiment_tracker import (
    ExperimentTracker,
    ExperimentConfig,
    Experiment,
    MetricRecord,
    HyperparameterSearch,
    TrainingProgressTracker,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tracker(temp_dir):
    return ExperimentTracker(temp_dir)


class TestExperimentTracker:
    def test_create_experiment(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        assert exp is not None
        assert exp.config.name == "test_exp"

    def test_log_metric(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        tracker.log_metric(exp.id, "loss", 0.5, 0)
        assert len(exp.metrics) == 1
        assert exp.metrics[0].name == "loss"

    def test_log_metrics(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        tracker.log_metrics(exp.id, {"loss": 0.5, "accuracy": 0.9}, 0)
        assert len(exp.metrics) == 2

    def test_complete_experiment(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        tracker.complete_experiment(exp.id)
        assert exp.status == "completed"
        assert exp.end_time is not None

    def test_list_experiments(self, tracker):
        tracker.create_experiment("exp1", "gpt2", 32768, 128, 1e-4, 4, 10)
        tracker.create_experiment("exp2", "gpt2", 16384, 64, 1e-3, 8, 5)
        experiments = tracker.list_experiments()
        assert len(experiments) == 2

    def test_get_best_epoch(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        tracker.log_metric(exp.id, "loss", 0.5, 0)
        tracker.log_metric(exp.id, "loss", 0.3, 1)
        tracker.log_metric(exp.id, "loss", 0.1, 2)
        best = tracker.get_best_epoch(exp.id, "loss", higher_is_better=False)
        assert best == (2, 0.1)

    def test_get_metrics_summary(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        tracker.log_metric(exp.id, "loss", 0.5, 0)
        tracker.log_metric(exp.id, "loss", 0.3, 1)
        summary = tracker.get_metrics_summary(exp.id)
        assert "loss" in summary
        assert summary["loss"]["min"] == 0.3


class TestHyperparameterSearch:
    def test_random_search(self, tracker):
        base_config = {"model_name": "gpt2", "epochs": 5}
        param_ranges = {
            "learning_rate": (1e-5, 1e-3),
            "dictionary_size": (8192, 32768),
        }
        experiments = HyperparameterSearch(tracker).random_search(base_config, param_ranges, 3)
        assert len(experiments) == 3


class TestTrainingProgressTracker:
    def test_progress_tracker(self, tracker):
        exp = tracker.create_experiment(
            name="test_exp",
            model_name="gpt2",
            dictionary_size=32768,
            top_k=128,
            learning_rate=1e-4,
            batch_size=4,
            epochs=10,
        )
        progress = TrainingProgressTracker(exp.id, tracker)
        progress.update_epoch(0)
        progress.log_epoch_metrics({"loss": 0.5})
        assert len(exp.metrics) == 1
