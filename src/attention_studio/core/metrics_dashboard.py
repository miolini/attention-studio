from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class MetricPoint:
    timestamp: float
    value: float
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    name: str
    description: str = ""
    unit: str = ""
    points: list[MetricPoint] = field(default_factory=list)
    max_points: int = 1000

    def add(self, value: float, step: int, metadata: dict[str, Any] | None = None) -> None:
        point = MetricPoint(
            timestamp=datetime.now().timestamp(),
            value=value,
            step=step,
            metadata=metadata or {},
        )
        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points.pop(0)

    def get_recent(self, n: int = 100) -> list[MetricPoint]:
        return self.points[-n:]

    def get_values(self) -> list[float]:
        return [p.value for p in self.points]

    def get_steps(self) -> list[int]:
        return [p.step for p in self.points]

    def get_latest(self) -> MetricPoint | None:
        return self.points[-1] if self.points else None

    def get_min(self) -> float | None:
        return min((p.value for p in self.points), default=None)

    def get_max(self) -> float | None:
        return max((p.value for p in self.points), default=None)

    def get_mean(self) -> float | None:
        if not self.points:
            return None
        return sum(p.value for p in self.points) / len(self.points)

    def clear(self) -> None:
        self.points.clear()


class MetricsDashboard:
    def __init__(self, max_points: int = 1000):
        self._metrics: dict[str, MetricSeries] = {}
        self._subscribers: list[Callable[[str, MetricPoint], None]] = []
        self._lock = Lock()
        self._max_points = max_points
        self._subscribers_lock = Lock()

    def create_metric(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> MetricSeries:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricSeries(
                    name=name,
                    description=description,
                    unit=unit,
                    max_points=self._max_points,
                )
            return self._metrics[name]

    def record(
        self,
        metric_name: str,
        value: float,
        step: int,
        description: str = "",
        unit: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = MetricSeries(
                    name=metric_name,
                    description=description,
                    unit=unit,
                    max_points=self._max_points,
                )

            series = self._metrics[metric_name]
            series.add(value, step, metadata)

        point = MetricPoint(
            timestamp=datetime.now().timestamp(),
            value=value,
            step=step,
            metadata=metadata or {},
        )

        with self._subscribers_lock:
            for subscriber in self._subscribers:
                with suppress(Exception):
                    subscriber(metric_name, point)

    def get_metric(self, name: str) -> MetricSeries | None:
        return self._metrics.get(name)

    def get_all_metrics(self) -> dict[str, MetricSeries]:
        return dict(self._metrics)

    def list_metrics(self) -> list[str]:
        return list(self._metrics.keys())

    def subscribe(self, callback: Callable[[str, MetricPoint], None]) -> None:
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, MetricPoint], None]) -> None:
        with self._subscribers_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def clear_metric(self, name: str) -> None:
        with self._lock:
            if name in self._metrics:
                self._metrics[name].clear()

    def clear_all(self) -> None:
        with self._lock:
            for series in self._metrics.values():
                series.clear()

    def get_summary(self) -> dict[str, dict[str, Any]]:
        summary = {}
        for name, series in self._metrics.items():
            summary[name] = {
                "description": series.description,
                "unit": series.unit,
                "count": len(series.points),
                "latest": series.get_latest().value if series.get_latest() else None,
                "min": series.get_min(),
                "max": series.get_max(),
                "mean": series.get_mean(),
            }
        return summary

    def export_json(self, path: Path | str) -> None:
        path = Path(path)
        data = {
            "exported_at": datetime.now().isoformat(),
            "metrics": {},
        }

        for name, series in self._metrics.items():
            data["metrics"][name] = {
                "description": series.description,
                "unit": series.unit,
                "points": [
                    {
                        "timestamp": p.timestamp,
                        "value": p.value,
                        "step": p.step,
                        "metadata": p.metadata,
                    }
                    for p in series.points
                ],
            }

        path.write_text(json.dumps(data, indent=2))

    def __len__(self) -> int:
        return len(self._metrics)

    def __getitem__(self, name: str) -> MetricSeries:
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found. Use create_metric() first.")
        return self._metrics[name]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __enter__(self) -> MetricsContext:
        return MetricsContext(self)

    def __exit__(self, *args: Any) -> None:
        pass


class MetricsContext:
    def __init__(self, dashboard: MetricsDashboard, prefix: str = ""):
        self._dashboard = dashboard
        self._prefix = prefix

    def record(
        self,
        metric_name: str,
        value: float,
        step: int,
        **kwargs: Any,
    ) -> None:
        full_name = f"{self._prefix}.{metric_name}" if self._prefix else metric_name
        self._dashboard.record(full_name, value, step, **kwargs)

    def __enter__(self) -> MetricsContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


_dashboard: MetricsDashboard | None = None


def get_metrics_dashboard() -> MetricsDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = MetricsDashboard()
    return _dashboard


def reset_metrics_dashboard() -> None:
    global _dashboard
    _dashboard = None
