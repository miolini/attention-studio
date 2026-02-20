from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ClickableNodeItem(QGraphicsEllipseItem):
    node_clicked = Signal(str)
    node_double_clicked = Signal(str)

    def __init__(self, node_id: str, x: float, y: float, width: float, height: float, node_data: dict[str, Any]):
        super().__init__(x, y, width, height)
        self.node_id = node_id
        self.node_data = node_data
        self._original_color = None
        self._is_selected = False
        self._is_highlighted = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def set_color(self, color: QColor):
        self._original_color = color
        from PySide6.QtGui import QBrush
        self.setBrush(QBrush(color))

    def set_selected(self, selected: bool):
        self._is_selected = selected
        if selected:
            self.setPen(QPen(QColor(255, 255, 0), 3))
        else:
            self.setPen(QPen(self._original_color.darker(150) if self._original_color else QColor(100, 100, 100), 2))

    def set_highlighted(self, highlighted: bool):
        self._is_highlighted = highlighted
        if highlighted:
            from PySide6.QtGui import QBrush
            self.setBrush(QBrush(QColor(255, 200, 0)))
        elif self._original_color:
            from PySide6.QtGui import QBrush
            self.setBrush(QBrush(self._original_color))

    def mousePressEvent(self, event):
        self.node_clicked.emit(self.node_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.node_double_clicked.emit(self.node_id)
        super().mouseDoubleClickEvent(event)


class HighlightableEdgeItem(QGraphicsLineItem):
    def __init__(self, source_id: str, target_id: str, weight: float, edge_type: str):
        super().__init__()
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.edge_type = edge_type
        self._original_color = None
        self._is_highlighted = False

    def set_original_color(self, color: QColor, width: float):
        self._original_color = color
        self.setPen(QPen(color, width))
        self._original_width = width

    def set_highlighted(self, highlighted: bool):
        self._is_highlighted = highlighted
        if highlighted:
            self.setPen(QPen(QColor(255, 200, 0), self._original_width * 2 + 1))
        else:
            self.setPen(QPen(self._original_color, self._original_width))


class NodeDetailsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._current_node_id = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._title_label = QLabel("Node Details")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #0e639c;")
        layout.addWidget(self._title_label)

        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(200)
        self._info_text.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                font-family: 'SF Mono', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._info_text)

        self._edges_label = QLabel("Connected Edges")
        self._edges_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(self._edges_label)

        self._edges_text = QTextEdit()
        self._edges_text.setReadOnly(True)
        self._edges_text.setMaximumHeight(150)
        self._edges_text.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                font-family: 'SF Mono', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self._edges_text)

        layout.addStretch()

    def show_node_details(self, node_id: str, node_data: dict[str, Any], edges: list[dict[str, Any]]):
        self._current_node_id = node_id

        info_lines = [
            f"Node ID: {node_id}",
            f"Type: {node_data.get('node_type', 'unknown')}",
            f"Layer: {node_data.get('layer', 'N/A')}",
            f"Position: {node_data.get('position', 'N/A')}",
        ]

        if node_data.get('feature_idx') is not None:
            info_lines.append(f"Feature: {node_data['feature_idx']}")

        if node_data.get('token'):
            info_lines.append(f"Token: '{node_data['token']}'")

        info_lines.append(f"Activation: {node_data.get('activation', 0):.6f}")

        if node_data.get('encoder_norm'):
            info_lines.append(f"Encoder Norm: {node_data['encoder_norm']:.4f}")

        if node_data.get('decoder_norm'):
            info_lines.append(f"Decoder Norm: {node_data['decoder_norm']:.4f}")

        self._info_text.setText("\n".join(info_lines))

        edge_lines = []
        for edge in edges[:20]:
            direction = "→" if edge.get('is_outgoing') else "←"
            weight = edge.get('weight', 0)
            edge_type = edge.get('edge_type', 'unknown')
            other_node = edge.get('other_node', 'unknown')
            edge_lines.append(f"{direction} {other_node}: {edge_type} ({weight:.4f})")

        if not edge_lines:
            edge_lines = ["No connected edges"]

        self._edges_text.setText("\n".join(edge_lines))

    def clear(self):
        self._current_node_id = None
        self._info_text.setText("Click a node to see details")
        self._edges_text.setText("")


class GraphFilterPanel(QWidget):
    filter_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Graph Filters")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #0e639c;")
        layout.addWidget(title)

        edge_weight_layout = QHBoxLayout()
        edge_weight_layout.addWidget(QLabel("Min Edge Weight:"))
        self._edge_weight_spin = QDoubleSpinBox()
        self._edge_weight_spin.setRange(0.0, 1.0)
        self._edge_weight_spin.setSingleStep(0.01)
        self._edge_weight_spin.setValue(0.01)
        self._edge_weight_spin.valueChanged.connect(self._on_filter_changed)
        edge_weight_layout.addWidget(self._edge_weight_spin)
        layout.addLayout(edge_weight_layout)

        node_type_layout = QHBoxLayout()
        node_type_layout.addWidget(QLabel("Node Type:"))
        self._node_type_combo = QComboBox()
        self._node_type_combo.addItems(["All", "embedding", "transcoder", "lorsa"])
        self._node_type_combo.currentTextChanged.connect(self._on_filter_changed)
        node_type_layout.addWidget(self._node_type_combo)
        layout.addLayout(node_type_layout)

        min_act_layout = QHBoxLayout()
        min_act_layout.addWidget(QLabel("Min Activation:"))
        self._min_activation_spin = QDoubleSpinBox()
        self._min_activation_spin.setRange(0.0, 10.0)
        self._min_activation_spin.setSingleStep(0.1)
        self._min_activation_spin.setValue(0.0)
        self._min_activation_spin.valueChanged.connect(self._on_filter_changed)
        min_act_layout.addWidget(self._min_activation_spin)
        layout.addLayout(min_act_layout)

        max_nodes_layout = QHBoxLayout()
        max_nodes_layout.addWidget(QLabel("Max Nodes:"))
        self._max_nodes_slider = QSlider()
        self._max_nodes_slider.setRange(10, 500)
        self._max_nodes_slider.setValue(100)
        self._max_nodes_slider.valueChanged.connect(self._on_filter_changed)
        max_nodes_layout.addWidget(self._max_nodes_slider)
        self._max_nodes_label = QLabel("100")
        max_nodes_layout.addWidget(self._max_nodes_label)
        layout.addLayout(max_nodes_layout)

        layout.addStretch()

    def _on_filter_changed(self):
        self._max_nodes_label.setText(str(self._max_nodes_slider.value()))
        filters = self.get_filters()
        self.filter_changed.emit(filters)

    def get_filters(self) -> dict[str, Any]:
        return {
            "min_edge_weight": self._edge_weight_spin.value(),
            "node_type": self._node_type_combo.currentText() if self._node_type_combo.currentText() != "All" else None,
            "min_activation": self._min_activation_spin.value(),
            "max_nodes": self._max_nodes_slider.value(),
        }

    def set_filters(self, filters: dict[str, Any]):
        if "min_edge_weight" in filters:
            self._edge_weight_spin.setValue(filters["min_edge_weight"])
        if "node_type" in filters:
            index = self._node_type_combo.findText(filters["node_type"])
            if index >= 0:
                self._node_type_combo.setCurrentIndex(index)
        if "min_activation" in filters:
            self._min_activation_spin.setValue(filters["min_activation"])
        if "max_nodes" in filters:
            self._max_nodes_slider.setValue(filters["max_nodes"])


class PathSelectionPanel(QWidget):
    find_paths_clicked = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._source_node = None
        self._target_node = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Path Finding")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #0e639c;")
        layout.addWidget(title)

        self._source_label = QLabel("Source: (click a node)")
        self._source_label.setStyleSheet("color: #4ec9b0;")
        layout.addWidget(self._source_label)

        self._target_label = QLabel("Target: (click a node)")
        self._target_label.setStyleSheet("color: #ce9178;")
        layout.addWidget(self._target_label)

        btn_layout = QHBoxLayout()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._clear_btn)

        self._find_btn = QPushButton("Find Paths")
        self._find_btn.clicked.connect(self._on_find_paths)
        self._find_btn.setEnabled(False)
        btn_layout.addWidget(self._find_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

    def set_source_node(self, node_id: str):
        self._source_node = node_id
        self._source_label.setText(f"Source: {node_id}")
        self._update_find_button()

    def set_target_node(self, node_id: str):
        self._target_node = node_id
        self._target_label.setText(f"Target: {node_id}")
        self._update_find_button()

    def _update_find_button(self):
        self._find_btn.setEnabled(self._source_node is not None and self._target_node is not None)

    def _on_clear(self):
        self._source_node = None
        self._target_node = None
        self._source_label.setText("Source: (click a node)")
        self._target_label.setText("Target: (click a node)")
        self._find_btn.setEnabled(False)

    def _on_find_paths(self):
        if self._source_node and self._target_node:
            self.find_paths_clicked.emit(self._source_node, self._target_node)

    def get_selection(self) -> tuple[str | None, str | None]:
        return self._source_node, self._target_node
