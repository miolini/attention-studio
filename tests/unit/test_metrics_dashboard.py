from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from attention_studio.core.metrics_dashboard import (
    MetricsContext,
    MetricsDashboard,
    MetricSeries,
    MetricPoint,
    get_metrics_dashboard,
    reset_metrics_dashboard,
)


class TestMetricPoint:
    def test_creation(self):
        point = MetricPoint(timestamp=1234567890.0, value=0.95, step=100)
        assert point.timestamp == 1234567890.0
        assert point.value == 0.95
        assert point.step == 100
        assert point.metadata == {}

    def test_with_metadata(self):
        point = MetricPoint(
            timestamp=1234567890.0,
            value=0.95,
            step=100,
            metadata={"layer": 5, "neuron": 10},
        )
        assert point.metadata["layer"] == 5


class TestMetricSeries:
    def test_default_values(self):
        series = MetricSeries(name="accuracy")
        assert series.name == "accuracy"
        assert series.description == ""
        assert series.unit == ""
        assert series.points == []
        assert series.max_points == 1000

    def test_add_points(self):
        series = MetricSeries(name="accuracy", max_points=10)
        series.add(0.5, 0)
        series.add(0.6, 1)
        series.add(0.7, 2)

        assert len(series.points) == 3
        assert series.get_latest().value == 0.7

    def test_max_points(self):
        series = MetricSeries(name="accuracy", max_points=3)
        for i in range(5):
            series.add(float(i), i)

        assert len(series.points) == 3
        assert series.points[0].value == 2.0

    def test_get_recent(self):
        series = MetricSeries(name="accuracy")
        for i in range(10):
            series.add(float(i), i)

        recent = series.get_recent(3)
        assert len(recent) == 3
        assert recent[0].value == 7.0

    def test_get_values(self):
        series = MetricSeries(name="accuracy")
        series.add(0.5, 0)
        series.add(0.6, 1)
        series.add(0.7, 2)

        values = series.get_values()
        assert values == [0.5, 0.6, 0.7]

    def test_statistics(self):
        series = MetricSeries(name="accuracy")
        series.add(1.0, 0)
        series.add(2.0, 1)
        series.add(3.0, 2)

        assert series.get_min() == 1.0
        assert series.get_max() == 3.0
        assert series.get_mean() == 2.0

    def test_clear(self):
        series = MetricSeries(name="accuracy")
        series.add(0.5, 0)
        series.add(0.6, 1)

        series.clear()
        assert len(series.points) == 0


class TestMetricsDashboard:
    @pytest.fixture(autouse=True)
    def reset_global(self):
        reset_metrics_dashboard()
        yield
        reset_metrics_dashboard()

    def test_create_metric(self):
        dashboard = MetricsDashboard()
        series = dashboard.create_metric("accuracy", "Model accuracy", "%")

        assert series.name == "accuracy"
        assert series.description == "Model accuracy"
        assert series.unit == "%"

    def test_record(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100, description="Model accuracy")

        series = dashboard.get_metric("accuracy")
        assert series is not None
        assert series.get_latest().value == 0.95

    def test_record_auto_creates_metric(self):
        dashboard = MetricsDashboard()
        dashboard.record("loss", 0.1, 50)

        assert "loss" in dashboard
        series = dashboard.get_metric("loss")
        assert series.get_latest().value == 0.1

    def test_get_metric(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100)

        series = dashboard.get_metric("accuracy")
        assert series is not None

    def test_get_metric_nonexistent(self):
        dashboard = MetricsDashboard()
        series = dashboard.get_metric("nonexistent")
        assert series is None

    def test_list_metrics(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100)
        dashboard.record("loss", 0.1, 100)

        metrics = dashboard.list_metrics()
        assert len(metrics) == 2

    def test_clear_metric(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100)
        dashboard.clear_metric("accuracy")

        series = dashboard.get_metric("accuracy")
        assert len(series.points) == 0

    def test_clear_all(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100)
        dashboard.record("loss", 0.1, 100)
        dashboard.clear_all()

        assert len(dashboard.list_metrics()) == 2
        for name in dashboard.list_metrics():
            series = dashboard.get_metric(name)
            assert len(series.points) == 0

    def test_summary(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100)
        dashboard.record("accuracy", 0.96, 101)
        dashboard.record("accuracy", 0.97, 102)

        summary = dashboard.get_summary()
        assert "accuracy" in summary
        assert summary["accuracy"]["count"] == 3
        assert summary["accuracy"]["latest"] == 0.97
        assert summary["accuracy"]["min"] == 0.95
        assert summary["accuracy"]["max"] == 0.97

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dashboard = MetricsDashboard()
            dashboard.record("accuracy", 0.95, 100)
            dashboard.record("loss", 0.1, 100)

            export_path = temp_dir / "metrics.json"
            dashboard.export_json(export_path)

            data = json.loads(export_path.read_text())
            assert "metrics" in data
            assert "accuracy" in data["metrics"]
            assert "loss" in data["metrics"]

    def test_subscribe(self):
        dashboard = MetricsDashboard()
        received = []

        def callback(name: str, point: MetricPoint) -> None:
            received.append((name, point.value))

        dashboard.subscribe(callback)
        dashboard.record("accuracy", 0.95, 100)

        assert len(received) == 1
        assert received[0] == ("accuracy", 0.95)

    def test_unsubscribe(self):
        dashboard = MetricsDashboard()
        received = []

        def callback(name: str, point: MetricPoint) -> None:
            received.append((name, point.value))

        dashboard.subscribe(callback)
        dashboard.unsubscribe(callback)
        dashboard.record("accuracy", 0.95, 100)

        assert len(received) == 0

    def test_context_manager(self):
        dashboard = MetricsDashboard()
        with dashboard as ctx:
            ctx.record("loss", 0.5, 0)
            ctx.record("loss", 0.4, 1)

        series = dashboard.get_metric("loss")
        assert len(series.points) == 2

    def test_prefixed_context(self):
        dashboard = MetricsDashboard()
        with MetricsContext(dashboard, prefix="train") as ctx:
            ctx.record("loss", 0.5, 0)

        series = dashboard.get_metric("train.loss")
        assert series is not None

    def test_getitem(self):
        dashboard = MetricsDashboard()
        dashboard.record("accuracy", 0.95, 100)

        series = dashboard["accuracy"]
        assert series.get_latest().value == 0.95

    def test_getitem_keyerror(self):
        dashboard = MetricsDashboard()
        with pytest.raises(KeyError):
            _ = dashboard["nonexistent"]


def test_get_metrics_dashboard_singleton():
    reset_metrics_dashboard()
    dashboard1 = get_metrics_dashboard()
    dashboard2 = get_metrics_dashboard()
    assert dashboard1 is dashboard2
    reset_metrics_dashboard()
