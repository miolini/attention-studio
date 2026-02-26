from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WidgetType(Enum):
    STAT = "stat"
    PROGRESS = "progress"
    CHART = "chart"
    GAUGE = "gauge"
    TEXT = "text"
    LIST = "list"


@dataclass
class DashboardWidget:
    id: str
    widget_type: WidgetType
    title: str
    value: Any = None
    unit: str = ""
    min_value: float = 0.0
    max_value: float = 100.0
    description: str = ""
    icon: str = ""
    color: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "widget_type": self.widget_type.value,
            "title": self.title,
            "value": self.value,
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardWidget:
        return cls(
            id=data["id"],
            widget_type=WidgetType(data.get("widget_type", "stat")),
            title=data.get("title", ""),
            value=data.get("value"),
            unit=data.get("unit", ""),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 100.0),
            description=data.get("description", ""),
            icon=data.get("icon", ""),
            color=data.get("color", ""),
            metadata=data.get("metadata", {}),
        )


class DashboardSection:
    def __init__(self, section_id: str, title: str, layout: str = "grid"):
        self.id = section_id
        self.title = title
        self.layout = layout
        self.widgets: list[DashboardWidget] = []

    def add_widget(self, widget: DashboardWidget) -> None:
        self.widgets.append(widget)

    def get_widget(self, widget_id: str) -> DashboardWidget | None:
        for widget in self.widgets:
            if widget.id == widget_id:
                return widget
        return None

    def remove_widget(self, widget_id: str) -> bool:
        for i, widget in enumerate(self.widgets):
            if widget.id == widget_id:
                self.widgets.pop(i)
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "layout": self.layout,
            "widgets": [w.to_dict() for w in self.widgets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardSection:
        section = cls(
            id=data["id"],
            title=data.get("title", ""),
            layout=data.get("layout", "grid"),
        )
        for w in data.get("widgets", []):
            section.widgets.append(DashboardWidget.from_dict(w))
        return section


class Dashboard:
    def __init__(self, dashboard_id: str, title: str):
        self.id = dashboard_id
        self.title = title
        self.sections: list[DashboardSection] = []
        self.metadata: dict[str, Any] = {}

    def add_section(self, section: DashboardSection) -> None:
        self.sections.append(section)

    def get_section(self, section_id: str) -> DashboardSection | None:
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def remove_section(self, section_id: str) -> bool:
        for i, section in enumerate(self.sections):
            if section.id == section_id:
                self.sections.pop(i)
                return True
        return False

    def get_all_widgets(self) -> list[DashboardWidget]:
        widgets = []
        for section in self.sections:
            widgets.extend(section.widgets)
        return widgets

    def get_widget(self, widget_id: str) -> DashboardWidget | None:
        for section in self.sections:
            widget = section.get_widget(widget_id)
            if widget:
                return widget
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dashboard:
        dashboard = cls(
            id=data["id"],
            title=data.get("title", ""),
        )
        dashboard.metadata = data.get("metadata", {})
        for s in data.get("sections", []):
            dashboard.sections.append(DashboardSection.from_dict(s))
        return dashboard


class DashboardManager:
    def __init__(self):
        self._dashboards: dict[str, Dashboard] = {}
        self._listeners: list[Callable[[str, str, Any], None]] = []
        self._create_default_dashboard()

    def _create_default_dashboard(self) -> None:
        dashboard = Dashboard("default", "Overview")

        overview = DashboardSection("overview", "Model Overview")
        overview.add_widget(DashboardWidget(
            id="model_name", widget_type=WidgetType.TEXT,
            title="Model", value="No model loaded", icon="model"
        ))
        overview.add_widget(DashboardWidget(
            id="device", widget_type=WidgetType.TEXT,
            title="Device", value="N/A", icon="device"
        ))
        overview.add_widget(DashboardWidget(
            id="parameters", widget_type=WidgetType.STAT,
            title="Parameters", value=0, unit="M", icon="params"
        ))
        dashboard.add_section(overview)

        metrics = DashboardSection("metrics", "Performance Metrics")
        metrics.add_widget(DashboardWidget(
            id="accuracy", widget_type=WidgetType.GAUGE,
            title="Accuracy", value=0.0, unit="%", min_value=0, max_value=100,
            color="#4ECDC4"
        ))
        metrics.add_widget(DashboardWidget(
            id="loss", widget_type=WidgetType.GAUGE,
            title="Loss", value=0.0, unit="", min_value=0, max_value=2,
            color="#FF6B6B"
        ))
        metrics.add_widget(DashboardWidget(
            id="throughput", widget_type=WidgetType.STAT,
            title="Throughput", value=0, unit="tok/s", icon="speed"
        ))
        dashboard.add_section(metrics)

        training = DashboardSection("training", "Training Status")
        training.add_widget(DashboardWidget(
            id="epoch", widget_type=WidgetType.PROGRESS,
            title="Epoch", value=0, unit="/ 0", min_value=0, max_value=100,
            description="Current training epoch"
        ))
        training.add_widget(DashboardWidget(
            id="step", widget_type=WidgetType.STAT,
            title="Step", value=0, unit="", icon="step"
        ))
        training.add_widget(DashboardWidget(
            id="eta", widget_type=WidgetType.TEXT,
            title="ETA", value="--:--", icon="time"
        ))
        dashboard.add_section(training)

        self._dashboards["default"] = dashboard

    def create_dashboard(self, dashboard_id: str, title: str) -> Dashboard:
        dashboard = Dashboard(dashboard_id, title)
        self._dashboards[dashboard_id] = dashboard
        return dashboard

    def get_dashboard(self, dashboard_id: str) -> Dashboard | None:
        return self._dashboards.get(dashboard_id)

    def delete_dashboard(self, dashboard_id: str) -> bool:
        if dashboard_id == "default":
            return False
        if dashboard_id in self._dashboards:
            del self._dashboards[dashboard_id]
            return True
        return False

    def list_dashboards(self) -> list[Dashboard]:
        return list(self._dashboards.values())

    def update_widget(self, dashboard_id: str, widget_id: str, value: Any) -> None:
        dashboard = self._dashboards.get(dashboard_id)
        if dashboard:
            widget = dashboard.get_widget(widget_id)
            if widget:
                widget.value = value
                self._notify_listeners(dashboard_id, widget_id, value)

    def subscribe(self, callback: Callable[[str, str, Any], None]) -> None:
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[str, str, Any], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, dashboard_id: str, widget_id: str, value: Any) -> None:
        for listener in self._listeners:
            with suppress(Exception):
                listener(dashboard_id, widget_id, value)


_dashboard_manager: DashboardManager | None = None


def get_dashboard_manager() -> DashboardManager:
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager


def get_default_dashboard() -> Dashboard:
    return get_dashboard_manager().get_dashboard("default")


def update_dashboard_widget(dashboard_id: str, widget_id: str, value: Any) -> None:
    get_dashboard_manager().update_widget(dashboard_id, widget_id, value)
