from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LayoutType(Enum):
    LAYERED = "layered"
    FORCE_DIRECTED = "force_directed"
    CIRCULAR = "circular"
    HIERARCHICAL = "hierarchical"
    RADIAL = "radial"


@dataclass
class NodePosition:
    x: float
    y: float


@dataclass
class GraphLayout:
    layout_type: LayoutType
    positions: dict[str, NodePosition] = field(default_factory=dict)


class GraphLayoutEngine:
    def __init__(self, graph: Any):
        self.graph = graph

    def layered_layout(
        self,
        layer_attr: str = "layer",
        spacing_x: float = 150.0,
        spacing_y: float = 80.0,
    ) -> dict[str, NodePosition]:
        nodes_by_layer: dict[int, list[str]] = defaultdict(list)

        for node in self.graph.nodes():
            layer = self.graph.nodes[node].get(layer_attr, 0)
            nodes_by_layer[layer].append(node)

        positions = {}
        sorted_layers = sorted(nodes_by_layer.keys())

        for layer_idx, layer in enumerate(sorted_layers):
            nodes = nodes_by_layer[layer]
            total_height = (len(nodes) - 1) * spacing_y
            start_y = -total_height / 2

            for i, node in enumerate(nodes):
                x = layer_idx * spacing_x
                y = start_y + i * spacing_y
                positions[node] = NodePosition(x, y)

        return positions

    def force_directed_layout(
        self,
        iterations: int = 100,
        repulsion: float = 5000.0,
        attraction: float = 0.01,
        damping: float = 0.9,
    ) -> dict[str, NodePosition]:
        positions = {}
        velocities = {}

        for node in self.graph.nodes():
            positions[node] = NodePosition(
                x=100 + (hash(node) % 500),
                y=100 + (hash(node + "y") % 500),
            )
            velocities[node] = NodePosition(x=0, y=0)

        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())

        for _ in range(iterations):
            for node_a in nodes:
                fx, fy = 0.0, 0.0

                for node_b in nodes:
                    if node_a == node_b:
                        continue

                    dx = positions[node_a].x - positions[node_b].x
                    dy = positions[node_a].y - positions[node_b].y
                    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)

                    force = repulsion / (dist * dist)
                    fx += (dx / dist) * force
                    fy += (dy / dist) * force

                for node_b in nodes:
                    if (node_a, node_b) in edges or (node_b, node_a) in edges:
                        dx = positions[node_b].x - positions[node_a].x
                        dy = positions[node_b].y - positions[node_a].y

                        fx += dx * attraction
                        fy += dy * attraction

                velocities[node_a].x = (velocities[node_a].x + fx) * damping
                velocities[node_a].y = (velocities[node_a].y + fy) * damping

            for node in nodes:
                positions[node].x += velocities[node].x
                positions[node].y += velocities[node].y

        return positions

    def circular_layout(
        self,
        radius: float = 200.0,
        center_x: float = 400.0,
        center_y: float = 300.0,
    ) -> dict[str, NodePosition]:
        nodes = list(self.graph.nodes())
        n = len(nodes)

        if n == 0:
            return {}

        positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * 3.14159 * i / n
            positions[node] = NodePosition(
                x=center_x + radius * (3.14159 / 2 - angle),
                y=center_y + radius * (3.14159 / 2 - angle),
            )

        return positions

    def hierarchical_layout(
        self,
        layer_attr: str = "layer",
        spacing_x: float = 150.0,
        spacing_y: float = 100.0,
    ) -> dict[str, NodePosition]:
        nodes_by_layer: dict[int, list[str]] = defaultdict(list)

        for node in self.graph.nodes():
            layer = self.graph.nodes[node].get(layer_attr, 0)
            nodes_by_layer[layer].append(node)

        positions = {}
        sorted_layers = sorted(nodes_by_layer.keys())

        max_nodes = max(len(v) for v in nodes_by_layer.values()) if nodes_by_layer else 1
        total_height = (max_nodes - 1) * spacing_y
        start_y = -total_height / 2

        for layer_idx, layer in enumerate(sorted_layers):
            nodes = nodes_by_layer[layer]
            layer_height = (len(nodes) - 1) * spacing_y
            layer_start_y = start_y + (total_height - layer_height) / 2

            for i, node in enumerate(nodes):
                x = layer_idx * spacing_x
                y = layer_start_y + i * spacing_y
                positions[node] = NodePosition(x, y)

        return positions

    def radial_layout(
        self,
        center_node: str | None = None,
        layer_attr: str = "layer",
        base_radius: float = 100.0,
        radius_increment: float = 80.0,
    ) -> dict[str, NodePosition]:
        if center_node and center_node in self.graph.nodes():
            root = center_node
        else:
            root = list(self.graph.nodes())[0] if self.graph.nodes() else None

        if not root:
            return {}

        layers: dict[int, list[str]] = defaultdict(list)
        visited = {root}
        current = [root]
        layer = 0

        while current:
            next_nodes = []
            for node in current:
                layers[layer].append(node)
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_nodes.append(neighbor)
            current = next_nodes
            layer += 1

        positions = {}
        center_x, center_y = 400.0, 300.0
        positions[root] = NodePosition(center_x, center_y)

        for layer_idx, nodes in layers.items():
            if layer_idx == 0:
                continue

            radius = base_radius + (layer_idx - 1) * radius_increment
            n = len(nodes)

            if n == 0:
                continue

            for i, node in enumerate(nodes):
                angle = 2 * 3.14159 * i / n
                positions[node] = NodePosition(
                    x=center_x + radius * (3.14159 / 2 - angle),
                    y=center_y + radius * (3.14159 / 2 - angle),
                )

        return positions

    def apply_layout(
        self,
        layout_type: LayoutType,
        **kwargs,
    ) -> dict[str, NodePosition]:
        if layout_type == LayoutType.LAYERED:
            return self.layered_layout(**kwargs)
        elif layout_type == LayoutType.FORCE_DIRECTED:
            return self.force_directed_layout(**kwargs)
        elif layout_type == LayoutType.CIRCULAR:
            return self.circular_layout(**kwargs)
        elif layout_type == LayoutType.HIERARCHICAL:
            return self.hierarchical_layout(**kwargs)
        elif layout_type == LayoutType.RADIAL:
            return self.radial_layout(**kwargs)
        else:
            return self.layered_layout()
