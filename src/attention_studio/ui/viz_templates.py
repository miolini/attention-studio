from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeStyleType(Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"


class EdgeStyleType(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"


class LayoutType(Enum):
    FORCE_DIRECTED = "force_directed"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    GRID = "grid"


@dataclass
class NodeStyle:
    shape: NodeStyleType = NodeStyleType.CIRCLE
    min_size: int = 10
    max_size: int = 30
    color: str = "#4A90D9"
    border_color: str = "#2C5F9E"
    border_width: int = 2
    label_visible: bool = True
    label_size: int = 12
    label_color: str = "#333333"


@dataclass
class EdgeStyle:
    style: EdgeStyleType = EdgeStyleType.SOLID
    width: float = 1.5
    color: str = "#999999"
    transparency: float = 0.6
    arrows: bool = True
    arrow_size: float = 0.5


@dataclass
class LayoutConfig:
    layout_type: LayoutType = LayoutType.FORCE_DIRECTED
    strength: float = 1.0
    distance: float = 100.0
    iterations: int = 100
    center: bool = True


@dataclass
class AnimationConfig:
    enabled: bool = True
    duration: int = 500
    easing: str = "ease-in-out"


@dataclass
class VisualizationStyle:
    background_color: str = "#FFFFFF"
    grid_visible: bool = False
    grid_color: str = "#EEEEEE"
    selection_color: str = "#FF6B6B"
    highlight_color: str = "#4ECDC4"


@dataclass
class VisualizationTemplate:
    name: str
    description: str = ""
    node_style: NodeStyle = field(default_factory=NodeStyle)
    edge_style: EdgeStyle = field(default_factory=EdgeStyle)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    style: VisualizationStyle = field(default_factory=VisualizationStyle)
    is_builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "node_style": {
                "shape": self.node_style.shape.value,
                "min_size": self.node_style.min_size,
                "max_size": self.node_style.max_size,
                "color": self.node_style.color,
                "border_color": self.node_style.border_color,
                "border_width": self.node_style.border_width,
                "label_visible": self.node_style.label_visible,
                "label_size": self.node_style.label_size,
                "label_color": self.node_style.label_color,
            },
            "edge_style": {
                "style": self.edge_style.style.value,
                "width": self.edge_style.width,
                "color": self.edge_style.color,
                "transparency": self.edge_style.transparency,
                "arrows": self.edge_style.arrows,
                "arrow_size": self.edge_style.arrow_size,
            },
            "layout": {
                "layout_type": self.layout.layout_type.value,
                "strength": self.layout.strength,
                "distance": self.layout.distance,
                "iterations": self.layout.iterations,
                "center": self.layout.center,
            },
            "animation": {
                "enabled": self.animation.enabled,
                "duration": self.animation.duration,
                "easing": self.animation.easing,
            },
            "style": {
                "background_color": self.style.background_color,
                "grid_visible": self.style.grid_visible,
                "grid_color": self.style.grid_color,
                "selection_color": self.style.selection_color,
                "highlight_color": self.style.highlight_color,
            },
            "is_builtin": self.is_builtin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualizationTemplate:
        ns = data.get("node_style", {})
        es = data.get("edge_style", {})
        ly = data.get("layout", {})
        an = data.get("animation", {})
        st = data.get("style", {})

        return cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            node_style=NodeStyle(
                shape=NodeStyleType(ns.get("shape", "circle")),
                min_size=ns.get("min_size", 10),
                max_size=ns.get("max_size", 30),
                color=ns.get("color", "#4A90D9"),
                border_color=ns.get("border_color", "#2C5F9E"),
                border_width=ns.get("border_width", 2),
                label_visible=ns.get("label_visible", True),
                label_size=ns.get("label_size", 12),
                label_color=ns.get("label_color", "#333333"),
            ),
            edge_style=EdgeStyle(
                style=EdgeStyleType(es.get("style", "solid")),
                width=es.get("width", 1.5),
                color=es.get("color", "#999999"),
                transparency=es.get("transparency", 0.6),
                arrows=es.get("arrows", True),
                arrow_size=es.get("arrow_size", 0.5),
            ),
            layout=LayoutConfig(
                layout_type=LayoutType(ly.get("layout_type", "force_directed")),
                strength=ly.get("strength", 1.0),
                distance=ly.get("distance", 100.0),
                iterations=ly.get("iterations", 100),
                center=ly.get("center", True),
            ),
            animation=AnimationConfig(
                enabled=an.get("enabled", True),
                duration=an.get("duration", 500),
                easing=an.get("easing", "ease-in-out"),
            ),
            style=VisualizationStyle(
                background_color=st.get("background_color", "#FFFFFF"),
                grid_visible=st.get("grid_visible", False),
                grid_color=st.get("grid_color", "#EEEEEE"),
                selection_color=st.get("selection_color", "#FF6B6B"),
                highlight_color=st.get("highlight_color", "#4ECDC4"),
            ),
            is_builtin=data.get("is_builtin", False),
        )


class VisualizationTemplateManager:
    def __init__(self, storage_dir: Path | None = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".attention_studio" / "viz_templates"
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, VisualizationTemplate] = {}
        self._load_builtins()
        self._load_user_templates()

    def _load_builtins(self) -> None:
        self._templates["default"] = VisualizationTemplate(
            name="Default",
            description="Default visualization style",
            is_builtin=True,
        )

        self._templates["dark"] = VisualizationTemplate(
            name="Dark Mode",
            description="Dark theme for presentations",
            node_style=NodeStyle(
                color="#61DAFB",
                border_color="#282C34",
                label_color="#FFFFFF",
            ),
            edge_style=EdgeStyle(color="#5C6370", transparency=0.4),
            style=VisualizationStyle(
                background_color="#1E1E1E",
                grid_color="#333333",
                selection_color="#E06C75",
                highlight_color="#98C379",
            ),
            is_builtin=True,
        )

        self._templates["academic"] = VisualizationTemplate(
            name="Academic",
            description="Clean style for papers and reports",
            node_style=NodeStyle(
                color="#FFFFFF",
                border_color="#000000",
                border_width=1,
                min_size=8,
                max_size=20,
            ),
            edge_style=EdgeStyle(
                color="#000000",
                transparency=0.3,
                arrows=True,
            ),
            layout=LayoutConfig(
                layout_type=LayoutType.HIERARCHICAL,
                iterations=200,
            ),
            animation=AnimationConfig(enabled=False),
            style=VisualizationStyle(
                background_color="#FFFFFF",
                grid_visible=True,
            ),
            is_builtin=True,
        )

        self._templates["presentation"] = VisualizationTemplate(
            name="Presentation",
            description="High contrast for presentations",
            node_style=NodeStyle(
                color="#FFD700",
                border_color="#FF8C00",
                min_size=20,
                max_size=40,
                label_size=14,
            ),
            edge_style=EdgeStyle(
                width=2.5,
                transparency=0.5,
                arrows=True,
                arrow_size=0.8,
            ),
            layout=LayoutConfig(
                layout_type=LayoutType.FORCE_DIRECTED,
                strength=0.5,
                distance=150,
            ),
            style=VisualizationStyle(
                background_color="#F5F5F5",
                selection_color="#FF4500",
                highlight_color="#32CD32",
            ),
            is_builtin=True,
        )

    def _load_user_templates(self) -> None:
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                template = VisualizationTemplate.from_dict(data)
                self._templates[template.name.lower().replace(" ", "_")] = template
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    def _save_template(self, template: VisualizationTemplate) -> None:
        path = self._storage_dir / f"{template.name.lower().replace(' ', '_')}.json"
        path.write_text(json.dumps(template.to_dict(), indent=2))

    def get_template(self, name: str) -> VisualizationTemplate | None:
        key = name.lower().replace(" ", "_")
        return self._templates.get(key)

    def list_templates(self, include_builtin: bool = True) -> list[VisualizationTemplate]:
        if include_builtin:
            return list(self._templates.values())
        return [t for t in self._templates.values() if not t.is_builtin]

    def list_builtin_templates(self) -> list[VisualizationTemplate]:
        return [t for t in self._templates.values() if t.is_builtin]

    def list_user_templates(self) -> list[VisualizationTemplate]:
        return [t for t in self._templates.values() if not t.is_builtin]

    def save_template(self, template: VisualizationTemplate) -> None:
        key = template.name.lower().replace(" ", "_")
        template.is_builtin = False
        self._templates[key] = template
        self._save_template(template)

    def delete_template(self, name: str) -> bool:
        key = name.lower().replace(" ", "_")
        if key in self._templates:
            template = self._templates[key]
            if template.is_builtin:
                return False
            del self._templates[key]
            path = self._storage_dir / f"{key}.json"
            if path.exists():
                path.unlink()
            return True
        return False

    def duplicate_template(self, source_name: str, new_name: str) -> VisualizationTemplate | None:
        source = self.get_template(source_name)
        if source is None:
            return None

        new_template = VisualizationTemplate(
            name=new_name,
            description=source.description,
            node_style=NodeStyle(
                shape=source.node_style.shape,
                min_size=source.node_style.min_size,
                max_size=source.node_style.max_size,
                color=source.node_style.color,
                border_color=source.node_style.border_color,
                border_width=source.node_style.border_width,
                label_visible=source.node_style.label_visible,
                label_size=source.node_style.label_size,
                label_color=source.node_style.label_color,
            ),
            edge_style=EdgeStyle(
                style=source.edge_style.style,
                width=source.edge_style.width,
                color=source.edge_style.color,
                transparency=source.edge_style.transparency,
                arrows=source.edge_style.arrows,
                arrow_size=source.edge_style.arrow_size,
            ),
            layout=LayoutConfig(
                layout_type=source.layout.layout_type,
                strength=source.layout.strength,
                distance=source.layout.distance,
                iterations=source.layout.iterations,
                center=source.layout.center,
            ),
            animation=AnimationConfig(
                enabled=source.animation.enabled,
                duration=source.animation.duration,
                easing=source.animation.easing,
            ),
            style=VisualizationStyle(
                background_color=source.style.background_color,
                grid_visible=source.style.grid_visible,
                grid_color=source.style.grid_color,
                selection_color=source.style.selection_color,
                highlight_color=source.style.highlight_color,
            ),
            is_builtin=False,
        )
        self.save_template(new_template)
        return new_template

    def export_template(self, name: str, path: Path | str) -> bool:
        template = self.get_template(name)
        if template is None:
            return False

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template.to_dict(), indent=2))
        return True

    def import_template(self, path: Path | str) -> VisualizationTemplate | None:
        path = Path(path)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            template = VisualizationTemplate.from_dict(data)
            template.is_builtin = False
            self.save_template(template)
            return template
        except (json.JSONDecodeError, KeyError, ValueError):
            return None


_template_manager: VisualizationTemplateManager | None = None


def get_template_manager() -> VisualizationTemplateManager:
    global _template_manager
    if _template_manager is None:
        _template_manager = VisualizationTemplateManager()
    return _template_manager


def get_template(name: str) -> VisualizationTemplate | None:
    return get_template_manager().get_template(name)
