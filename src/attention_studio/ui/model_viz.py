"""
Model visualization widget with force-directed layout.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from attention_studio.ui.graphics_view import InteractiveGraphicsView


class ElementType(Enum):
    TENSOR = "tensor"
    OPERATOR = "operator"


@dataclass
class Block:
    id: int
    label: str
    x: float = 0.0
    y: float = 0.0
    width: float = 60.0
    height: float = 24.0
    element_type: ElementType = ElementType.TENSOR
    layer_order: int = 0
    tensor_shape: tuple | None = None

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class Connection:
    source_id: int
    target_id: int
    color: QColor = field(default_factory=lambda: QColor(100, 100, 120))


class ForceDirectedLayout:
    """Incremental force-directed layout optimizer."""

    def __init__(self, blocks: list[Block], connections: list[Connection]):
        self.blocks = {b.id: b for b in blocks}
        self.connections = connections
        self.velocities: dict[int, tuple[float, float]] = {b.id: (0.0, 0.0) for b in blocks}

        self.repulsion_strength = 8000.0
        self.attraction_strength = 0.008
        self.gravity_strength = 0.1
        self.damping = 0.85
        self.min_distance = 80.0
        self.max_speed = 50.0
        self.max_position = 800.0
        self.collision_strength = 20000.0

    def step(self) -> dict:
        """Perform one iteration of force-directed layout."""
        forces: dict[int, tuple[float, float]] = dict.fromkeys(self.blocks, (0.0, 0.0))

        for bid1 in self.blocks:
            for bid2 in self.blocks:
                if bid1 == bid2:
                    continue

                b1 = self.blocks[bid1]
                b2 = self.blocks[bid2]

                dx = b2.center_x - b1.center_x
                dy = b2.center_y - b1.center_y
                dist = max(1.0, math.sqrt(dx * dx + dy * dy))

                force = self.repulsion_strength / (dist * dist)
                fx = -dx / dist * force
                fy = -dy / dist * force

                f1x, f1y = forces[bid1]
                f2x, f2y = forces[bid2]
                forces[bid1] = (f1x + fx, f1y + fy)
                forces[bid2] = (f2x - fx, f2y - fy)

        for bid1 in self.blocks:
            for bid2 in self.blocks:
                if bid1 >= bid2:
                    continue

                b1 = self.blocks[bid1]
                b2 = self.blocks[bid2]

                overlap_x = (b1.width / 2 + b2.width / 2) - abs(b2.center_x - b1.center_x)
                overlap_y = (b1.height / 2 + b2.height / 2) - abs(b2.center_y - b1.center_y)

                if overlap_x > 0 and overlap_y > 0:
                    overlap = min(overlap_x, overlap_y)
                    dx = b2.center_x - b1.center_x
                    dy = b2.center_y - b1.center_y

                    if abs(dx) < 0.01:
                        dx = 1.0
                    if abs(dy) < 0.01:
                        dy = 1.0

                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 0.01:
                        dist = 0.01

                    force = self.collision_strength * (1 + overlap * 0.5)
                    fx = -dx / dist * force
                    fy = -dy / dist * force

                    f1x, f1y = forces[bid1]
                    f2x, f2y = forces[bid2]
                    forces[bid1] = (f1x + fx, f1y + fy)
                    forces[bid2] = (f2x - fx, f2y - fy)

        connected = set()
        for conn in self.connections:
            src = self.blocks.get(conn.source_id)
            tgt = self.blocks.get(conn.target_id)
            if not src or not tgt:
                continue

            connected.add((conn.source_id, conn.target_id))

            dx = tgt.center_x - src.center_x
            dy = tgt.center_y - src.center_y
            dist = max(1.0, math.sqrt(dx * dx + dy * dy))

            force = dist * self.attraction_strength
            fx = dx / dist * force
            fy = dy / dist * force

            sfx, sfy = forces[conn.source_id]
            tfx, tfy = forces[conn.target_id]
            forces[conn.source_id] = (sfx + fx, sfy + fy)
            forces[conn.target_id] = (tfx - fx, tfy - fy)

        center_x = sum(b.center_x for b in self.blocks.values()) / len(self.blocks)
        center_y = sum(b.center_y for b in self.blocks.values()) / len(self.blocks)

        for bid, block in self.blocks.items():
            dx = center_x - block.center_x
            dy = center_y - block.center_y

            fx, fy = forces[bid]
            forces[bid] = (
                fx + dx * self.gravity_strength,
                fy + dy * self.gravity_strength,
            )

        for bid in self.blocks:
            fx, fy = forces[bid]
            vx, vy = self.velocities[bid]

            vx = (vx + fx) * self.damping
            vy = (vy + fy) * self.damping

            speed = math.sqrt(vx * vx + vy * vy)
            if speed > self.max_speed:
                vx = vx / speed * self.max_speed
                vy = vy / speed * self.max_speed

            self.velocities[bid] = (vx, vy)

            block.x = max(-self.max_position, min(self.max_position, block.x + vx))
            block.y = max(-self.max_position, min(self.max_position, block.y + vy))

        return {b.id: (b.x, b.y) for b in self.blocks.values()}

    def run(self, iterations: int = 100) -> list[dict]:
        """Run optimization for given iterations, yielding positions incrementally."""
        results = []
        for _ in range(iterations):
            pos = self.step()
            results.append(pos)
        return results


class LayoutWorker(QObject):
    progress = Signal(dict)
    finished = Signal()

    def __init__(self, blocks: list[Block], connections: list[Connection]):
        super().__init__()
        self._blocks = blocks
        self._connections = connections
        self._layout = ForceDirectedLayout(blocks, connections)

    def run(self):
        for i, pos in enumerate(self._layout.run(200)):
            if i % 10 == 0:
                self.progress.emit(pos)
        self.finished.emit()


class ModelVisualizationWidget(QFrame):
    layer_selected = Signal(int)

    def __init__(self, parent=None, use_threaded_optimization=True):
        super().__init__(parent)
        self._zoom = 1.0
        self._blocks: list[Block] = []
        self._connections: list[Connection] = []
        self._optimizer_thread = None
        self._optimizer_worker = None
        self._use_threaded_optimization = use_threaded_optimization
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setFixedHeight(40)
        toolbar.setStyleSheet("background-color: #252526; border-bottom: 1px solid #3c3c3c;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 0, 12, 0)

        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(1, 5000)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(150)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        toolbar_layout.addWidget(QLabel("Zoom:"))
        toolbar_layout.addWidget(self._zoom_slider)

        toolbar_layout.addStretch()

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedWidth(30)
        self._zoom_in_btn.setToolTip("Zoom In")
        self._zoom_in_btn.clicked.connect(self._on_zoom_in)
        toolbar_layout.addWidget(self._zoom_in_btn)

        self._zoom_out_btn = QPushButton("-")
        self._zoom_out_btn.setFixedWidth(30)
        self._zoom_out_btn.setToolTip("Zoom Out")
        self._zoom_out_btn.clicked.connect(self._on_zoom_out)
        toolbar_layout.addWidget(self._zoom_out_btn)

        self._reset_btn = QPushButton("Fit")
        self._reset_btn.setToolTip("Fit to View")
        self._reset_btn.clicked.connect(self._on_fit_view)
        toolbar_layout.addWidget(self._reset_btn)

        main_layout.addWidget(toolbar)

        self._scene = QGraphicsScene()
        self._view = InteractiveGraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform)
        self._view.setBackgroundBrush(QColor(18, 18, 18))
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.zoom_changed.connect(self._on_view_zoom_changed)
        main_layout.addWidget(self._view)

        info_bar = QFrame()
        info_bar.setStyleSheet("background-color: #252526; border-top: 1px solid #3c3c3c;")
        info_bar.setFixedHeight(28)
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(12, 0, 12, 0)

        self._info_label = QLabel("No model loaded")
        self._info_label.setStyleSheet("color: #808080; font-size: 11px;")
        info_layout.addWidget(self._info_label)

        info_layout.addStretch()

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #808080; font-size: 11px;")
        info_layout.addWidget(self._stats_label)

        main_layout.addWidget(info_bar)

        self._draw_empty_state()

    def _on_zoom_changed(self, value):
        self._zoom = value / 100.0
        self._view.set_zoom(self._zoom)

    def _on_zoom_in(self):
        new_value = min(300, self._zoom_slider.value() + 25)
        self._zoom_slider.setValue(new_value)

    def _on_zoom_out(self):
        new_value = max(25, self._zoom_slider.value() - 25)
        self._zoom_slider.setValue(new_value)

    def _on_fit_view(self):
        self._view.fit_in_view()
        self._update_slider_from_view()

    def _on_view_zoom_changed(self, zoom_level):
        self._zoom_slider.setValue(int(zoom_level * 100))
        self._zoom = zoom_level

    def _update_slider_from_view(self):
        current_zoom = self._view.get_zoom()
        self._zoom_slider.setValue(int(current_zoom * 100))
        self._zoom = current_zoom

    def _draw_empty_state(self):
        self._scene.clear()
        text = self._scene.addText("Load a model to see visualization")
        text.setDefaultTextColor(QColor(128, 128, 128))
        font = QFont("SF Pro Display", 16)
        text.setFont(font)
        text.setPos(200, 200)

    def _create_model_blocks(self, config, num_layers: int) -> tuple[list[Block], list[Connection]]:
        """Create blocks and connections from model config."""
        hidden_size: int = cast(int, getattr(config, "n_embd", None) or getattr(config, "hidden_size", 768))
        num_heads: int = cast(int, getattr(config, "n_head", None) or getattr(config, "num_attention_heads", 12))
        vocab_size: int = cast(int, getattr(config, "vocab_size", 50257))
        head_dim = hidden_size // num_heads

        blocks = []
        connections = []

        block_id = 0

        tensor_color = QColor(46, 204, 113)
        operator_color = QColor(155, 89, 182)
        mlp_color = QColor(230, 126, 34)
        layer_norm_color = QColor(241, 196, 15)
        output_color = QColor(231, 76, 60)

        tensor_width = 70
        tensor_height = 25
        op_width = 50
        op_height = 30

        block_id += 1
        input_block = Block(
            id=block_id,
            label="input",
            element_type=ElementType.TENSOR,
            layer_order=0,
            width=tensor_width,
            height=tensor_height,
            tensor_shape=(1, vocab_size),
        )
        blocks.append(input_block)

        block_id += 1
        wte_block = Block(
            id=block_id,
            label="WTE",
            element_type=ElementType.TENSOR,
            layer_order=1,
            width=tensor_width,
            height=tensor_height,
            tensor_shape=(vocab_size, hidden_size),
        )
        blocks.append(wte_block)
        connections.append(Connection(input_block.id, wte_block.id, tensor_color))

        block_id += 1
        wpe_block = Block(
            id=block_id,
            label="WPE",
            element_type=ElementType.TENSOR,
            layer_order=2,
            width=tensor_width,
            height=tensor_height,
            tensor_shape=(config.n_positions, hidden_size),
        )
        blocks.append(wpe_block)
        connections.append(Connection(input_block.id, wpe_block.id, tensor_color))

        block_id += 1
        embed_add = Block(
            id=block_id,
            label="+",
            element_type=ElementType.OPERATOR,
            layer_order=3,
            width=op_width,
            height=op_height,
        )
        blocks.append(embed_add)
        connections.append(Connection(wte_block.id, embed_add.id, operator_color))
        connections.append(Connection(wpe_block.id, embed_add.id, operator_color))

        block_id += 1
        hidden_block = Block(
            id=block_id,
            label="hidden",
            element_type=ElementType.TENSOR,
            layer_order=4,
            width=tensor_width,
            height=tensor_height,
            tensor_shape=(1, hidden_size),
        )
        blocks.append(hidden_block)
        connections.append(Connection(embed_add.id, hidden_block.id, tensor_color))

        prev_block = hidden_block

        for layer_idx in range(num_layers):
            layer_base = layer_idx * 100 + 10

            block_id += 1
            ln1 = Block(
                id=block_id,
                label=f"ln{layer_idx}",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 10,
                width=op_width,
                height=op_height,
            )
            blocks.append(ln1)
            connections.append(Connection(prev_block.id, ln1.id, layer_norm_color))

            block_id += 1
            qkv_block = Block(
                id=block_id,
                label=f"qkv{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 20,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size, hidden_size * 3),
            )
            blocks.append(qkv_block)
            connections.append(Connection(ln1.id, qkv_block.id, tensor_color))

            block_id += 1
            q_block = Block(
                id=block_id,
                label=f"Q{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 30,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size, head_dim * num_heads),
            )
            blocks.append(q_block)
            connections.append(Connection(qkv_block.id, q_block.id, tensor_color))

            block_id += 1
            k_block = Block(
                id=block_id,
                label=f"K{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 31,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size, head_dim * num_heads),
            )
            blocks.append(k_block)
            connections.append(Connection(qkv_block.id, k_block.id, tensor_color))

            block_id += 1
            v_block = Block(
                id=block_id,
                label=f"V{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 32,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size, head_dim * num_heads),
            )
            blocks.append(v_block)
            connections.append(Connection(qkv_block.id, v_block.id, tensor_color))

            block_id += 1
            qk_matmul = Block(
                id=block_id,
                label="QK^T",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 40,
                width=op_width,
                height=op_height,
            )
            blocks.append(qk_matmul)
            connections.append(Connection(q_block.id, qk_matmul.id, operator_color))
            connections.append(Connection(k_block.id, qk_matmul.id, operator_color))

            block_id += 1
            softmax_block = Block(
                id=block_id,
                label="softmax",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 50,
                width=op_width,
                height=op_height,
            )
            blocks.append(softmax_block)
            connections.append(Connection(qk_matmul.id, softmax_block.id, operator_color))

            block_id += 1
            attn_matmul = Block(
                id=block_id,
                label="attn",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 60,
                width=op_width,
                height=op_height,
            )
            blocks.append(attn_matmul)
            connections.append(Connection(softmax_block.id, attn_matmul.id, operator_color))
            connections.append(Connection(v_block.id, attn_matmul.id, operator_color))

            block_id += 1
            proj_block = Block(
                id=block_id,
                label=f"proj{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 70,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size, hidden_size),
            )
            blocks.append(proj_block)
            connections.append(Connection(attn_matmul.id, proj_block.id, tensor_color))

            block_id += 1
            attn_add = Block(
                id=block_id,
                label="+",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 80,
                width=op_width,
                height=op_height,
            )
            blocks.append(attn_add)
            connections.append(Connection(ln1.id, attn_add.id, operator_color))
            connections.append(Connection(proj_block.id, attn_add.id, operator_color))

            block_id += 1
            attn_output = Block(
                id=block_id,
                label=f"out{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 90,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(1, hidden_size),
            )
            blocks.append(attn_output)
            connections.append(Connection(attn_add.id, attn_output.id, tensor_color))

            block_id += 1
            ln2 = Block(
                id=block_id,
                label=f"ln{layer_idx}",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 100,
                width=op_width,
                height=op_height,
            )
            blocks.append(ln2)
            connections.append(Connection(attn_output.id, ln2.id, layer_norm_color))

            block_id += 1
            fc1_block = Block(
                id=block_id,
                label=f"fc1{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 110,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size, hidden_size * 4),
            )
            blocks.append(fc1_block)
            connections.append(Connection(ln2.id, fc1_block.id, mlp_color))

            block_id += 1
            gelu_block = Block(
                id=block_id,
                label="GELU",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 120,
                width=op_width,
                height=op_height,
            )
            blocks.append(gelu_block)
            connections.append(Connection(fc1_block.id, gelu_block.id, mlp_color))

            block_id += 1
            fc2_block = Block(
                id=block_id,
                label=f"fc2{layer_idx}",
                element_type=ElementType.TENSOR,
                layer_order=layer_base + 130,
                width=tensor_width,
                height=tensor_height,
                tensor_shape=(hidden_size * 4, hidden_size),
            )
            blocks.append(fc2_block)
            connections.append(Connection(gelu_block.id, fc2_block.id, mlp_color))

            block_id += 1
            mlp_add = Block(
                id=block_id,
                label="+",
                element_type=ElementType.OPERATOR,
                layer_order=layer_base + 140,
                width=op_width,
                height=op_height,
            )
            blocks.append(mlp_add)
            connections.append(Connection(ln2.id, mlp_add.id, mlp_color))
            connections.append(Connection(fc2_block.id, mlp_add.id, mlp_color))

            prev_block = mlp_add

        block_id += 1
        final_ln = Block(
            id=block_id,
            label="ln_f",
            element_type=ElementType.OPERATOR,
            layer_order=1000,
            width=op_width,
            height=op_height,
        )
        blocks.append(final_ln)
        connections.append(Connection(prev_block.id, final_ln.id, layer_norm_color))

        block_id += 1
        lm_head_block = Block(
            id=block_id,
            label="lm_head",
            element_type=ElementType.TENSOR,
            layer_order=1010,
            width=tensor_width,
            height=tensor_height,
            tensor_shape=(hidden_size, vocab_size),
        )
        blocks.append(lm_head_block)
        connections.append(Connection(final_ln.id, lm_head_block.id, output_color))

        block_id += 1
        logits_block = Block(
            id=block_id,
            label="logits",
            element_type=ElementType.TENSOR,
            layer_order=1020,
            width=tensor_width,
            height=tensor_height,
            tensor_shape=(1, vocab_size),
        )
        blocks.append(logits_block)
        connections.append(Connection(lm_head_block.id, logits_block.id, output_color))

        block_id += 1
        softmax_out = Block(
            id=block_id,
            label="softmax",
            element_type=ElementType.OPERATOR,
            layer_order=1030,
            width=op_width,
            height=op_height,
        )
        blocks.append(softmax_out)
        connections.append(Connection(logits_block.id, softmax_out.id, output_color))

        return blocks, connections

    def update_model(self, model_manager, layer_indices):
        if self._optimizer_thread is not None:
            try:
                if self._optimizer_thread.isRunning():
                    self._optimizer_thread.quit()
                    self._optimizer_thread.wait(1000)
            except RuntimeError:
                pass
            self._optimizer_thread = None
            self._optimizer_worker = None

        self._scene.clear()
        self._blocks = []
        self._connections = []

        config = model_manager.model.config
        num_layers = len(layer_indices)

        self._all_tensors = []
        for name, param in model_manager.model.named_parameters():
            self._all_tensors.append({
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype).replace("torch.", ""),
                "numel": param.numel(),
            })

        self._num_layers = num_layers
        self._num_heads = getattr(config, "n_head", None) or getattr(config, "num_attention_heads", 12)
        self._hidden_size = getattr(config, "n_embd", None) or getattr(config, "hidden_size", 768)

        self._blocks, self._connections = self._create_model_blocks(config, num_layers)

        max_layer_order = max(b.layer_order for b in self._blocks) if self._blocks else 1
        for block in self._blocks:
            x_pos = 50 + (block.layer_order / max_layer_order) * 400 * num_layers
            y_pos = 50 + (block.layer_order % 20) * 30
            block.x = max(-800, min(2500, x_pos))
            block.y = max(-800, min(2500, y_pos))

        if self._use_threaded_optimization:
            self._start_threaded_optimization()
        else:
            self._run_sync_optimization()

    def _start_threaded_optimization(self):
        self._optimizer_thread = QThread()
        self._optimizer_worker = LayoutWorker(self._blocks, self._connections)
        self._optimizer_worker.moveToThread(self._optimizer_thread)

        self._optimizer_thread.started.connect(self._optimizer_worker.run)
        self._optimizer_worker.progress.connect(self._on_optimizer_progress)
        self._optimizer_worker.finished.connect(self._on_optimizer_finished)
        self._optimizer_worker.finished.connect(self._optimizer_thread.quit)
        self._optimizer_thread.finished.connect(self._optimizer_worker.deleteLater)
        self._optimizer_thread.finished.connect(self._optimizer_thread.deleteLater)

        self._optimizer_thread.start()

    def _run_sync_optimization(self):
        layout = ForceDirectedLayout(self._blocks, self._connections)

        for positions in layout.run(200):
            self._apply_positions(positions)

        self._resolve_overlaps()

        self._render()
        self._finalize()

    def _resolve_overlaps(self, iterations: int = 50):
        for _ in range(iterations):
            any_moved = False

            for i, b1 in enumerate(self._blocks):
                for b2 in self._blocks[i + 1:]:
                    ox = max(0, min(b1.x + b1.width, b2.x + b2.width) - max(b1.x, b2.x))
                    oy = max(0, min(b1.y + b1.height, b2.y + b2.height) - max(b1.y, b2.y))

                    if ox > 0 and oy > 0:
                        dx = b2.center_x - b1.center_x
                        dy = b2.center_y - b1.center_y

                        if abs(dx) < 0.01:
                            dx = 1.0
                        if abs(dy) < 0.01:
                            dy = 1.0

                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < 0.01:
                            dist = 0.01

                        sep_x = ox / 2 + 2
                        sep_y = oy / 2 + 2

                        nx = dx / dist
                        ny = dy / dist

                        b1.x -= nx * sep_x
                        b1.y -= ny * sep_y
                        b2.x += nx * sep_x
                        b2.y += ny * sep_y

                        any_moved = True

            if not any_moved:
                break

    def _on_optimizer_progress(self, positions):
        self._apply_positions(positions)
        self._render()

    def _apply_positions(self, positions):
        for block in self._blocks:
            if block.id in positions:
                block.x = positions[block.id][0]
                block.y = positions[block.id][1]

    def _on_optimizer_finished(self):
        self._finalize()

    def _render(self):
        self._scene.clear()

        for block in self._blocks:
            if block.element_type == ElementType.OPERATOR:
                rect = self._scene.addEllipse(
                    block.x, block.y, block.width, block.height
                )
                rect.setBrush(QBrush(QColor(155, 89, 182, 150)))
                rect.setPen(QPen(QColor(155, 89, 182), 2))
                rect.setZValue(10)
            else:
                rect = self._scene.addRect(block.x, block.y, block.width, block.height)
                rect.setBrush(QBrush(QColor(46, 204, 113, 100)))
                rect.setPen(QPen(QColor(46, 204, 113), 1.5))
                rect.setZValue(10)

            text = self._scene.addText(block.label)
            text.setDefaultTextColor(QColor(220, 220, 230))
            text.setFont(QFont("SF Mono", 7))
            text.setPos(block.x + 2, block.y + 2)
            text.setZValue(20)

        block_map = {b.id: b for b in self._blocks}

        def _line_intersects_rect(x1: float, y1: float, x2: float, y2: float, block: Block, margin: float = 10.0) -> bool:
            expanded_x = block.x - margin
            expanded_y = block.y - margin
            expanded_right = block.right + margin
            expanded_bottom = block.bottom + margin

            if min(x1, x2) > expanded_right or max(x1, x2) < expanded_x:
                return False
            return not (min(y1, y2) > expanded_bottom or max(y1, y2) < expanded_y)

        def _get_blocking_blocks(x1: float, y1: float, x2: float, y2: float, src_id: int, tgt_id: int) -> list[Block]:
            blocking = []
            for block in self._blocks:
                if block.id == src_id or block.id == tgt_id:
                    continue
                if _line_intersects_rect(x1, y1, x2, y2, block):
                    blocking.append(block)
            return blocking

        def _compute_arrow_path(
            src_x: float, src_y: float, tgt_x: float, tgt_y: float,
            src_block: Block, tgt_block: Block
        ) -> list[tuple[float, float]]:
            direct_blocking = _get_blocking_blocks(src_x, src_y, tgt_x, tgt_y, src_block.id, tgt_block.id)

            if not direct_blocking:
                return [(src_x, src_y), (tgt_x, tgt_y)]

            src_cx = src_block.center_x
            src_cy = src_block.center_y
            tgt_cx = tgt_block.center_x

            mid_x = (src_cx + tgt_cx) / 2

            all_blocking_y = [b.center_y for b in direct_blocking]
            avg_blocking_y = sum(all_blocking_y) / len(all_blocking_y)

            if src_cy < avg_blocking_y:
                detour_y = min(b.y for b in self._blocks) - 50
            else:
                detour_y = max(b.bottom for b in self._blocks) + 50

            return [(src_x, src_y), (mid_x, detour_y), (tgt_x, tgt_y)]

        for conn in self._connections:
            src = block_map.get(conn.source_id)
            tgt = block_map.get(conn.target_id)
            if not src or not tgt:
                continue

            start_x = src.right
            start_y = src.center_y
            end_x = tgt.left
            end_y = tgt.center_y

            min_y = min(b.y for b in self._blocks)
            max_y = max(b.bottom for b in self._blocks)

            route_y = min_y - 200 if start_y < (min_y + max_y) / 2 else max_y + 200

            mid_x = (start_x + end_x) / 2
            path_points = [(start_x, start_y), (mid_x, route_y), (end_x, end_y)]

            path = QPainterPath()
            path.moveTo(path_points[0][0], path_points[0][1])

            if len(path_points) == 2:
                px1, py1 = path_points[0]
                px2, py2 = path_points[1]
                ctrl = max(20, abs(px2 - px1) * 0.35)
                path.cubicTo(px1 + ctrl, py1, px2 - ctrl, py2, px2, py2)
            else:
                for i in range(1, len(path_points) - 1):
                    curr = path_points[i]
                    path.lineTo(curr[0], curr[1])
                px, py = path_points[-1]
                path.lineTo(px, py)

            line = self._scene.addPath(path)
            line.setPen(QPen(conn.color, 1.5))
            line.setZValue(5)

            arrow_path = QPainterPath()
            arrow_path.moveTo(end_x, end_y)
            arrow_path.lineTo(end_x - 6, end_y - 4)
            arrow_path.lineTo(end_x - 6, end_y + 4)
            arrow_path.closeSubpath()

            arrow = self._scene.addPath(arrow_path)
            arrow.setBrush(QBrush(conn.color))
            arrow.setZValue(6)

    def _finalize(self):
        if not self._blocks:
            return

        self._scene.setSceneRect(-50000, -50000, 100000, 100000)

        total_params = sum(t["numel"] for t in self._all_tensors)
        self._info_label.setText(
            f"Layers: {self._num_layers} | Heads: {self._num_heads} | Hidden: {self._hidden_size:,}"
        )
        self._stats_label.setText(
            f"{total_params/1e6:.1f}M params | {len(self._all_tensors)} tensors | {int(self._zoom*100)}% zoom"
        )

    def wait_for_optimization(self, timeout_ms=30000):
        from PySide6.QtCore import QEventLoop, QTimer

        try:
            if self._optimizer_thread is None:
                return
            if not self._optimizer_thread.isRunning():
                return
        except RuntimeError:
            return

        loop = QEventLoop()
        self._optimizer_thread.finished.connect(loop.quit)
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()
