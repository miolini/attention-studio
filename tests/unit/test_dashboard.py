from __future__ import annotations

import pytest

from attention_studio.ui.dashboard import (
    Dashboard,
    DashboardManager,
    DashboardSection,
    DashboardWidget,
    WidgetType,
    get_dashboard_manager,
    get_default_dashboard,
    update_dashboard_widget,
)


class TestDashboardWidget:
    def test_default_values(self):
        widget = DashboardWidget(id="test", widget_type=WidgetType.STAT, title="Test")
        assert widget.id == "test"
        assert widget.widget_type == WidgetType.STAT
        assert widget.title == "Test"
        assert widget.value is None

    def test_to_dict(self):
        widget = DashboardWidget(
            id="test", widget_type=WidgetType.GAUGE,
            title="Accuracy", value=95.5, unit="%"
        )
        data = widget.to_dict()
        assert data["id"] == "test"
        assert data["value"] == 95.5

    def test_from_dict(self):
        data = {
            "id": "test",
            "widget_type": "gauge",
            "title": "Accuracy",
            "value": 95.5,
            "unit": "%",
        }
        widget = DashboardWidget.from_dict(data)
        assert widget.id == "test"
        assert widget.widget_type == WidgetType.GAUGE
        assert widget.value == 95.5


class TestDashboardSection:
    def test_add_widget(self):
        section = DashboardSection("sec1", "Section 1")
        widget = DashboardWidget(id="w1", widget_type=WidgetType.STAT, title="Test")
        section.add_widget(widget)
        assert len(section.widgets) == 1

    def test_get_widget(self):
        section = DashboardSection("sec1", "Section 1")
        widget = DashboardWidget(id="w1", widget_type=WidgetType.STAT, title="Test")
        section.add_widget(widget)
        assert section.get_widget("w1") is not None

    def test_remove_widget(self):
        section = DashboardSection("sec1", "Section 1")
        widget = DashboardWidget(id="w1", widget_type=WidgetType.STAT, title="Test")
        section.add_widget(widget)
        assert section.remove_widget("w1") is True
        assert len(section.widgets) == 0


class TestDashboard:
    def test_add_section(self):
        dashboard = Dashboard("dash1", "Dashboard 1")
        section = DashboardSection("sec1", "Section 1")
        dashboard.add_section(section)
        assert len(dashboard.sections) == 1

    def test_get_section(self):
        dashboard = Dashboard("dash1", "Dashboard 1")
        section = DashboardSection("sec1", "Section 1")
        dashboard.add_section(section)
        assert dashboard.get_section("sec1") is not None

    def test_get_all_widgets(self):
        dashboard = Dashboard("dash1", "Dashboard 1")
        section = DashboardSection("sec1", "Section 1")
        section.add_widget(DashboardWidget(id="w1", widget_type=WidgetType.STAT, title="Test1"))
        section.add_widget(DashboardWidget(id="w2", widget_type=WidgetType.STAT, title="Test2"))
        dashboard.add_section(section)

        widgets = dashboard.get_all_widgets()
        assert len(widgets) == 2

    def test_to_dict(self):
        dashboard = Dashboard("dash1", "Dashboard 1")
        data = dashboard.to_dict()
        assert data["id"] == "dash1"
        assert data["title"] == "Dashboard 1"


class TestDashboardManager:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        import attention_studio.ui.dashboard as db
        db._dashboard_manager = None
        yield
        db._dashboard_manager = None

    def test_default_dashboard_created(self):
        manager = DashboardManager()
        dashboards = manager.list_dashboards()
        assert len(dashboards) >= 1

    def test_create_dashboard(self):
        manager = DashboardManager()
        dashboard = manager.create_dashboard("custom", "Custom Dashboard")
        assert dashboard.id == "custom"
        assert dashboard.title == "Custom Dashboard"

    def test_get_dashboard(self):
        manager = DashboardManager()
        dashboard = manager.get_dashboard("default")
        assert dashboard is not None

    def test_delete_dashboard(self):
        manager = DashboardManager()
        manager.create_dashboard("to_delete", "To Delete")
        assert manager.delete_dashboard("to_delete") is True
        assert manager.get_dashboard("to_delete") is None

    def test_delete_default_fails(self):
        manager = DashboardManager()
        assert manager.delete_dashboard("default") is False

    def test_update_widget(self):
        manager = DashboardManager()
        manager.update_widget("default", "accuracy", 95.5)
        widget = manager.get_dashboard("default").get_widget("accuracy")
        assert widget.value == 95.5

    def test_subscribe(self):
        manager = DashboardManager()
        received = []

        def callback(dash_id: str, widget_id: str, value) -> None:
            received.append((dash_id, widget_id, value))

        manager.subscribe(callback)
        manager.update_widget("default", "accuracy", 99.0)
        assert len(received) == 1
        assert received[0] == ("default", "accuracy", 99.0)

    def test_unsubscribe(self):
        manager = DashboardManager()
        received = []

        def callback(dash_id: str, widget_id: str, value) -> None:
            received.append((dash_id, widget_id, value))

        manager.subscribe(callback)
        manager.unsubscribe(callback)
        manager.update_widget("default", "accuracy", 99.0)
        assert len(received) == 0


def test_get_dashboard_manager_singleton():
    import attention_studio.ui.dashboard as db
    db._dashboard_manager = None
    mgr1 = get_dashboard_manager()
    mgr2 = get_dashboard_manager()
    assert mgr1 is mgr2
    db._dashboard_manager = None


def test_get_default_dashboard():
    import attention_studio.ui.dashboard as db
    db._dashboard_manager = None
    dashboard = get_default_dashboard()
    assert dashboard.id == "default"
    db._dashboard_manager = None


def test_update_dashboard_widget():
    import attention_studio.ui.dashboard as db
    db._dashboard_manager = None
    update_dashboard_widget("default", "accuracy", 88.0)
    dashboard = get_default_dashboard()
    widget = dashboard.get_widget("accuracy")
    assert widget.value == 88.0
    db._dashboard_manager = None
