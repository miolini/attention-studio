from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FeatureHeatmapWidget(QWidget):
    feature_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._activations = None
        self._tokens = []
        self._feature_indices = []
        self._min_val = 0.0
        self._max_val = 1.0
        self._hover_pos = None
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    def set_data(
        self,
        activations: torch.Tensor | np.ndarray,
        tokens: list[str],
        feature_indices: list[int] | None = None,
    ):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()

        self._activations = activations
        self._tokens = tokens
        self._feature_indices = feature_indices or list(range(activations.shape[0] if len(activations.shape) > 0 else 0))

        if activations.size > 0:
            self._min_val = float(np.min(activations))
            self._max_val = float(np.max(activations))
            if self._max_val == self._min_val:
                self._max_val = self._min_val + 1.0

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._activations is None or self._activations.size == 0:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data")
            return

        num_features, num_positions = self._activations.shape
        if num_features == 0 or num_positions == 0:
            return

        cell_width = max(20, min(60, (self.width() - 100) // num_positions))
        cell_height = max(15, min(30, (self.height() - 50) // num_features))

        margin_left = 80
        margin_top = 40

        for i, token in enumerate(self._tokens[:num_positions]):
            x = margin_left + i * cell_width
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(
                x, margin_top - 10, cell_width, 20,
                Qt.AlignmentFlag.AlignCenter,
                token[:8]
            )

        for fi, feat_idx in enumerate(self._feature_indices[:num_features]):
            y = margin_top + fi * cell_height
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(
                5, y, margin_left - 10, cell_height,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"F{feat_idx}"
            )

        for fi in range(min(num_features, len(self._feature_indices))):
            for pi in range(min(num_positions, len(self._tokens))):
                val = self._activations[fi, pi]
                normalized = (val - self._min_val) / (self._max_val - self._min_val)

                if val >= 0:
                    r = int(255 * normalized)
                    g = int(100 * (1 - normalized))
                    b = int(100 * (1 - normalized))
                else:
                    r = int(50 * (1 - normalized))
                    g = int(50 * (1 - normalized))
                    b = int(255 * normalized)

                color = QColor(min(255, r), min(255, g), min(255, b))
                x = margin_left + pi * cell_width
                y = margin_top + fi * cell_height

                painter.fillRect(x, y, cell_width - 1, cell_height - 1, color)

        painter.setPen(QColor(100, 100, 100))
        for i in range(num_positions + 1):
            x = margin_left + i * cell_width
            painter.drawLine(x, margin_top, x, margin_top + num_features * cell_height)

        for i in range(num_features + 1):
            y = margin_top + i * cell_height
            painter.drawLine(margin_left, y, margin_left + num_positions * cell_width, y)

    def mouseMoveEvent(self, event):
        if self._activations is None:
            return

        num_features, num_positions = self._activations.shape
        cell_width = max(20, min(60, (self.width() - 100) // num_positions))
        cell_height = max(15, min(30, (self.height() - 50) // num_features))
        margin_left = 80
        margin_top = 40

        x = event.position().x()
        y = event.position().y()

        pi = int((x - margin_left) / cell_width)
        fi = int((y - margin_top) / cell_height)

        if 0 <= pi < num_positions and 0 <= fi < num_features:
            val = self._activations[fi, pi]
            token = self._tokens[pi] if pi < len(self._tokens) else "?"
            feat_idx = self._feature_indices[fi] if fi < len(self._feature_indices) else fi
            self.setToolTip(f"Feature {feat_idx}, Token '{token}': {val:.4f}")

    def mousePressEvent(self, event):
        if self._activations is None:
            return

        num_features, num_positions = self._activations.shape
        cell_width = max(20, min(60, (self.width() - 100) // num_positions))
        cell_height = max(15, min(30, (self.height() - 50) // num_features))
        margin_left = 80
        margin_top = 40

        x = event.position().x()
        y = event.position().y()

        pi = int((x - margin_left) / cell_width)
        fi = int((y - margin_top) / cell_height)

        if 0 <= pi < num_positions and 0 <= fi < num_features:
            feat_idx = self._feature_indices[fi] if fi < len(self._feature_indices) else fi
            self.feature_clicked.emit(feat_idx, pi)


class FeatureHeatmapPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Top Features:"))
        self._top_k_spin = QSpinBox()
        self._top_k_spin.setRange(5, 100)
        self._top_k_spin.setValue(20)
        controls_layout.addWidget(self._top_k_spin)

        controls_layout.addWidget(QLabel("Sort By:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Activation", "Variance", "Max Position"])
        controls_layout.addWidget(self._sort_combo)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        self._heatmap = FeatureHeatmapWidget()
        scroll.setWidget(self._heatmap)
        layout.addWidget(scroll)

        self._colorbar_label = QLabel("Activation: Blue (negative) → Red (positive)")
        self._colorbar_label.setStyleSheet("color: #808080; font-size: 10px;")
        layout.addWidget(self._colorbar_label)

    def set_feature_data(
        self,
        features: list[dict[str, Any]],
        tokens: list[str],
        activations_by_position: dict[int, list[float]] | None = None,
    ):
        if not features or not tokens:
            self._heatmap.set_data(np.array([]), [], [])
            return

        top_k = self._top_k_spin.value()
        sort_by = self._sort_combo.currentText()

        if sort_by == "Activation":
            sorted_features = sorted(features, key=lambda x: abs(x.get("activation", 0)), reverse=True)
        elif sort_by == "Variance":
            sorted_features = sorted(features, key=lambda x: x.get("variance", 0), reverse=True)
        else:
            sorted_features = sorted(features, key=lambda x: x.get("max_activation", 0), reverse=True)

        top_features = sorted_features[:top_k]
        feature_indices = [f.get("idx", i) for i, f in enumerate(top_features)]

        if activations_by_position:
            activations = np.zeros((len(feature_indices), len(tokens)))
            for fi, feat_idx in enumerate(feature_indices):
                if feat_idx in activations_by_position:
                    acts = activations_by_position[feat_idx]
                    for pi in range(min(len(acts), len(tokens))):
                        activations[fi, pi] = acts[pi]
        else:
            activations = np.zeros((len(feature_indices), len(tokens)))
            for fi, feat in enumerate(top_features):
                activation = feat.get("activation", 0)
                activations[fi, :] = activation

        self._heatmap.set_data(activations, tokens, feature_indices)

    def get_top_k(self) -> int:
        return self._top_k_spin.value()

    def get_sort_method(self) -> str:
        return self._sort_combo.currentText()


class CircuitVisualizationWidget(QWidget):
    circuit_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._circuits = {}
        self._min_size = 600
        self.setMinimumSize(self._min_size, 400)
        self.setMouseTracking(True)

    def set_circuits(self, circuits: dict[str, list[Any]]):
        self._circuits = circuits
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if not self._circuits:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No circuits detected")
            return

        circuit_types = list(self._circuits.keys())
        if not circuit_types:
            return

        num_types = len(circuit_types)
        max_width = self.width() - 40
        max_height = self.height() - 80

        box_width = min(200, max_width // max(1, num_types))
        box_height = max_height - 20

        margin_x = 20
        margin_y = 60

        colors = [
            QColor(46, 204, 113),
            QColor(52, 152, 219),
            QColor(155, 89, 182),
            QColor(241, 196, 15),
            QColor(230, 126, 34),
            QColor(231, 76, 60),
            QColor(26, 188, 156),
            QColor(149, 165, 166),
        ]

        for i, circuit_type in enumerate(circuit_types):
            circuits = self._circuits[circuit_type]
            if not circuits:
                continue

            x = margin_x + i * (box_width + 20)
            y = margin_y
            color = colors[i % len(colors)]

            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 30))
            painter.drawRoundedRect(x, y, box_width, box_height, 10, 10)

            painter.setPen(color)
            font = painter.font()
            font.setBold(True)
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(x + 10, y - 20, circuit_type.replace("_", " ").title())

            font.setBold(False)
            font.setPointSize(9)
            painter.setFont(font)
            painter.setPen(QColor(200, 200, 200))

            strength = circuits[0].strength if hasattr(circuits[0], 'strength') else 0
            painter.drawText(x + 10, y + 25, f"Strength: {strength:.3f}")

            features = circuits[0].features if hasattr(circuits[0], 'features') else []
            painter.drawText(x + 10, y + 45, f"Features: {len(features)}")

            max_show = min(8, len(features))
            for j, feat in enumerate(features[:max_show]):
                if isinstance(feat, tuple) and len(feat) >= 2:
                    layer, feat_idx = feat[0], feat[1]
                    painter.drawText(x + 10, y + 70 + j * 18, f"L{layer}F{feat_idx}")

            if len(features) > max_show:
                painter.drawText(x + 10, y + 70 + max_show * 18, f"... +{len(features) - max_show} more")

    def mousePressEvent(self, event):
        if not self._circuits:
            return

        circuit_types = list(self._circuits.keys())
        box_width = min(200, (self.width() - 40) // max(1, len(circuit_types)))
        margin_x = 20

        x = event.position().x()
        for i, circuit_type in enumerate(circuit_types):
            box_x = margin_x + i * (box_width + 20)
            if box_x <= x <= box_x + box_width:
                self.circuit_clicked.emit(circuit_type)
                break
