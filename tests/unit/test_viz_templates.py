from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from attention_studio.ui.viz_templates import (
    AnimationConfig,
    EdgeStyle,
    EdgeStyleType,
    LayoutConfig,
    LayoutType,
    NodeStyle,
    NodeStyleType,
    VisualizationStyle,
    VisualizationTemplate,
    VisualizationTemplateManager,
    get_template,
    get_template_manager,
)


class TestNodeStyle:
    def test_default_values(self):
        style = NodeStyle()
        assert style.shape == NodeStyleType.CIRCLE
        assert style.min_size == 10
        assert style.max_size == 30
        assert style.color == "#4A90D9"


class TestEdgeStyle:
    def test_default_values(self):
        style = EdgeStyle()
        assert style.style == EdgeStyleType.SOLID
        assert style.width == 1.5
        assert style.color == "#999999"
        assert style.transparency == 0.6


class TestLayoutConfig:
    def test_default_values(self):
        config = LayoutConfig()
        assert config.layout_type == LayoutType.FORCE_DIRECTED
        assert config.strength == 1.0
        assert config.iterations == 100


class TestAnimationConfig:
    def test_default_values(self):
        config = AnimationConfig()
        assert config.enabled is True
        assert config.duration == 500
        assert config.easing == "ease-in-out"


class TestVisualizationStyle:
    def test_default_values(self):
        style = VisualizationStyle()
        assert style.background_color == "#FFFFFF"
        assert style.grid_visible is False
        assert style.selection_color == "#FF6B6B"


class TestVisualizationTemplate:
    def test_default_values(self):
        template = VisualizationTemplate(name="Test")
        assert template.name == "Test"
        assert template.description == ""
        assert isinstance(template.node_style, NodeStyle)
        assert isinstance(template.edge_style, EdgeStyle)

    def test_to_dict(self):
        template = VisualizationTemplate(name="Test", description="A test")
        data = template.to_dict()
        assert data["name"] == "Test"
        assert data["description"] == "A test"
        assert "node_style" in data

    def test_from_dict(self):
        data = {
            "name": "Test",
            "description": "A test",
            "node_style": {"shape": "square", "color": "#FF0000"},
            "edge_style": {"style": "dashed"},
            "layout": {"layout_type": "circular"},
            "animation": {"enabled": False},
            "style": {"background_color": "#000000"},
        }
        template = VisualizationTemplate.from_dict(data)
        assert template.name == "Test"
        assert template.node_style.shape == NodeStyleType.SQUARE
        assert template.node_style.color == "#FF0000"
        assert template.edge_style.style == EdgeStyleType.DASHED
        assert template.layout.layout_type == LayoutType.CIRCULAR
        assert template.animation.enabled is False
        assert template.style.background_color == "#000000"


class TestVisualizationTemplateManager:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_loads_builtins(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        templates = manager.list_templates()
        assert len(templates) >= 4
        assert any(t.name == "Default" for t in templates)
        assert any(t.name == "Dark Mode" for t in templates)
        assert any(t.name == "Academic" for t in templates)

    def test_get_template(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        template = manager.get_template("Default")
        assert template is not None
        assert template.name == "Default"

    def test_get_template_case_insensitive(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        template = manager.get_template("default")
        assert template is not None

    def test_get_nonexistent(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        template = manager.get_template("nonexistent")
        assert template is None

    def test_list_builtin(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        builtin = manager.list_builtin_templates()
        assert all(t.is_builtin for t in builtin)

    def test_save_template(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        template = VisualizationTemplate(
            name="Custom Template",
            description="My custom style",
            is_builtin=False,
        )
        manager.save_template(template)

        manager2 = VisualizationTemplateManager(temp_dir)
        loaded = manager2.get_template("Custom Template")
        assert loaded is not None
        assert loaded.description == "My custom style"

    def test_delete_user_template(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        template = VisualizationTemplate(name="To Delete", is_builtin=False)
        manager.save_template(template)

        result = manager.delete_template("To Delete")
        assert result is True

        loaded = manager.get_template("To Delete")
        assert loaded is None

    def test_delete_builtin_fails(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        result = manager.delete_template("Default")
        assert result is False

    def test_duplicate_template(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        new_template = manager.duplicate_template("Default", "My Copy")
        assert new_template is not None
        assert new_template.name == "My Copy"
        assert new_template.is_builtin is False

    def test_duplicate_nonexistent(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        result = manager.duplicate_template("Nonexistent", "New")
        assert result is None

    def test_export_template(self, temp_dir):
        manager = VisualizationTemplateManager(temp_dir)
        path = temp_dir / "export.json"
        result = manager.export_template("Default", path)

        assert result is True
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "Default"

    def test_import_template(self, temp_dir):
        template = VisualizationTemplate(
            name="Imported",
            description="Imported template",
        )
        export_path = temp_dir / "import_source.json"
        export_path.write_text(json.dumps(template.to_dict()))

        manager = VisualizationTemplateManager(temp_dir)
        imported = manager.import_template(export_path)

        assert imported is not None
        assert imported.name == "Imported"

    def test_singleton(self):
        import attention_studio.ui.viz_templates as vt
        vt._template_manager = None
        mgr1 = get_template_manager()
        mgr2 = get_template_manager()
        assert mgr1 is mgr2
        vt._template_manager = None


def test_get_template_helper():
    import attention_studio.ui.viz_templates as vt
    vt._template_manager = None
    template = get_template("Default")
    assert template is not None
    assert template.name == "Default"
    vt._template_manager = None
