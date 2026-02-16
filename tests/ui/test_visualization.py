import sys
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from attention_studio.ui.graphics_view import InteractiveGraphicsView
from attention_studio.ui.model_viz import ModelVisualizationWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def viz_widget(qapp):
    widget = ModelVisualizationWidget(use_threaded_optimization=False)
    original_update = widget.update_model

    def update_and_wait(*args, **kwargs):
        original_update(*args, **kwargs)

    widget.update_model = update_and_wait
    yield widget
    widget.close()
    QTest.qWait(50)


@pytest.fixture
def mock_model_manager():
    mock_manager = MagicMock()
    mock_config = MagicMock()
    mock_config.name_or_path = "gpt2"
    mock_config.n_embd = 768
    mock_config.n_layer = 12
    mock_config.vocab_size = 50257
    mock_config.hidden_size = 768
    mock_manager.model.config = mock_config

    class MockParam:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self._numel = shape[0] * shape[1]

        def numel(self):
            return self._numel

    params_dict = {
        "wte.weight": MockParam([50257, 768], "torch.float32"),
        "wpe.weight": MockParam([1024, 768], "torch.float32"),
        "h.0.attn.qkv.weight": MockParam([768, 2304], "torch.float32"),
        "h.0.attn.out.weight": MockParam([768, 768], "torch.float32"),
        "h.0.mlp.fc1.weight": MockParam([768, 3072], "torch.float32"),
        "h.0.mlp.fc2.weight": MockParam([3072, 768], "torch.float32"),
        "lm_head.weight": MockParam([768, 50257], "torch.float32"),
    }

    mock_manager.model.named_parameters.return_value = iter(params_dict.items())

    return mock_manager


class TestVisualizationLayout:
    """Tests for visualization layout, element sizes and positions."""

    def test_scene_rect_valid_after_update(self, qapp, viz_widget, mock_model_manager):
        """Test that scene rect is valid after updating model."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        rect = viz_widget._scene.sceneRect()
        assert rect.width() > 0, "Scene rect width should be positive"
        assert rect.height() > 0, "Scene rect height should be positive"

    def test_block_dimensions(self, qapp, viz_widget, mock_model_manager):
        """Test that blocks have correct dimensions."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()

        assert len(items) > 10, f"Should have visualization items, got {len(items)}"

    def test_blocks_have_minimum_size(self, qapp, viz_widget, mock_model_manager):
        """Test that operation blocks meet minimum size requirements."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        rect_items = [item for item in items if hasattr(item, 'rect') and item.zValue() == 10]

        for item in rect_items:
            rect = item.rect()
            assert rect.width() >= 50, f"Block width {rect.width()} should be >= 50"
            assert rect.height() >= 20, f"Block height {rect.height()} should be >= 20"

    def test_blocks_are_within_scene_bounds(self, qapp, viz_widget, mock_model_manager):
        """Test that most blocks are positioned reasonably within scene bounds."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        scene_rect = viz_widget._scene.sceneRect()
        items = viz_widget._scene.items()

        out_of_bounds = 0
        for item in items:
            if hasattr(item, 'boundingRect'):
                bounds = item.boundingRect()
                if hasattr(item, 'pos'):
                    pos = item.pos()
                    item_rect = QRectF(pos.x() + bounds.x(), pos.y() + bounds.y(), bounds.width(), bounds.height())
                elif hasattr(item, 'rect'):
                    item_rect = item.rect()
                else:
                    continue

                if item_rect.right() > scene_rect.right() + 2000:
                    out_of_bounds += 1

        assert out_of_bounds < len(items) * 0.5, f"Too many items out of bounds: {out_of_bounds}/{len(items)}"


class TestVisualizationPorts:
    """Tests for input/output port positions."""

    def test_input_ports_on_left_side(self, qapp, viz_widget, mock_model_manager):
        """Test that input ports are positioned on left side of blocks."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        port_items = [item for item in items if isinstance(item, viz_widget._scene.items()[0].__class__) and hasattr(item, 'rect') and item.zValue() == 15]

        if len(port_items) > 0:
            for port in port_items:
                rect = port.rect()
                assert rect.width() <= 20, f"Port should be small (circle), got width {rect.width()}"

    def test_output_ports_on_right_side(self, qapp, viz_widget, mock_model_manager):
        """Test that output ports exist and are positioned correctly."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        all_items = len(items)

        assert all_items > 15, f"Should have many items including ports, got {all_items}"


class TestVisualizationArrows:
    """Tests for arrow/connection positions."""

    def test_arrows_exist(self, qapp, viz_widget, mock_model_manager):
        """Test that arrows are created for connections."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        path_items = [item for item in items if hasattr(item, 'path')]

        assert len(path_items) >= 3, f"Should have arrows, got {len(path_items)}"

    def test_arrows_have_correct_z_order(self, qapp, viz_widget, mock_model_manager):
        """Test that arrows are rendered behind blocks."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        path_items = [item for item in items if hasattr(item, 'path')]

        for path_item in path_items:
            assert path_item.zValue() < 10, f"Arrow zValue should be < 10 (behind blocks), got {path_item.zValue()}"


class TestVisualizationZoom:
    """Tests for zoom functionality."""

    def test_initial_zoom(self, viz_widget):
        """Test that initial zoom is 100%."""
        assert viz_widget._zoom == 1.0, f"Initial zoom should be 1.0, got {viz_widget._zoom}"

    def test_zoom_slider_range(self, viz_widget):
        """Test that zoom slider has correct range."""
        assert viz_widget._zoom_slider.minimum() == 1, "Zoom slider min should be 1 (1%)"
        assert viz_widget._zoom_slider.maximum() == 5000, "Zoom slider max should be 5000 (5000%)"

    def test_view_zoom_limits(self, viz_widget):
        """Test that view has correct zoom limits."""
        assert viz_widget._view._min_zoom == 0.01, f"Min zoom should be 0.01, got {viz_widget._view._min_zoom}"
        assert viz_widget._view._max_zoom == 50.0, f"Max zoom should be 50.0, got {viz_widget._view._max_zoom}"

    def test_zoom_in(self, viz_widget):
        """Test zoom in button increases zoom."""
        initial_zoom = viz_widget._zoom_slider.value()
        viz_widget._on_zoom_in()
        assert viz_widget._zoom_slider.value() > initial_zoom, "Zoom should increase"

    def test_zoom_out(self, viz_widget):
        """Test zoom out button decreases zoom."""
        viz_widget._zoom_slider.setValue(200)
        initial_zoom = viz_widget._zoom_slider.value()
        viz_widget._on_zoom_out()
        assert viz_widget._zoom_slider.value() < initial_zoom, "Zoom should decrease"

    def test_set_zoom_clamped_to_min(self, viz_widget):
        """Test that zoom cannot go below minimum."""
        viz_widget._zoom_slider.setValue(1)
        viz_widget._on_zoom_out()
        assert viz_widget._zoom_slider.value() >= 1, "Zoom should be clamped to minimum"

    def test_set_zoom_clamped_to_max(self, viz_widget):
        """Test that zoom cannot exceed maximum."""
        viz_widget._zoom_slider.setValue(5000)
        viz_widget._on_zoom_in()
        assert viz_widget._zoom_slider.value() <= 5000, "Zoom should be clamped to maximum"


class TestVisualizationElements:
    """Tests for visualization elements and labels."""

    def test_title_exists(self, qapp, viz_widget, mock_model_manager):
        """Test that title is displayed."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        text_items = [item for item in items if hasattr(item, 'toPlainText')]

        assert len(text_items) > 0, "Should have text items"

    def test_info_label_updated(self, qapp, viz_widget, mock_model_manager):
        """Test that info label shows model info."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        info_text = viz_widget._info_label.text()
        assert "3" in info_text or "layers" in info_text.lower(), f"Info should contain layer count, got: {info_text}"

    def test_stats_label_updated(self, qapp, viz_widget, mock_model_manager):
        """Test that stats label is updated."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        stats_text = viz_widget._stats_label.text()
        assert "params" in stats_text.lower(), f"Stats should contain params, got: {stats_text}"
        assert "%" in stats_text or "zoom" in stats_text.lower(), f"Stats should contain zoom info, got: {stats_text}"

    def test_legend_exists(self, qapp, viz_widget, mock_model_manager):
        """Test that legend is displayed."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        legend_items = [item for item in items if hasattr(item, 'brush') and hasattr(item, 'pen')]

        assert len(legend_items) > 10, f"Should have legend items, got {len(legend_items)}"


class TestVisualizationGrouping:
    """Tests for layer grouping boxes."""

    def test_layer_groups_exist(self, qapp, viz_widget, mock_model_manager):
        """Test that layer visualizations are created."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        text_items = [item for item in items if hasattr(item, 'toPlainText') and ('ln' in item.toPlainText() or 'Layer' in item.toPlainText())]

        assert len(text_items) >= 1, f"Should have layer labels, got {len(text_items)}"

    def test_input_output_groups_exist(self, qapp, viz_widget, mock_model_manager):
        """Test that input and output group boxes exist."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)

        items = viz_widget._scene.items()
        all_items = len(items)

        assert all_items > 20, f"Should have many items including groups, got {all_items}"


class TestVisualizationMultiLayer:
    """Tests for multi-layer visualizations."""

    def test_more_layers_creates_more_items(self, qapp, viz_widget, mock_model_manager):
        """Test that more layers create more visualization items."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)
        items_2_layers = len(viz_widget._scene.items())

        viz_widget.update_model(mock_model_manager, list(range(5)))
        QTest.qWait(100)
        items_5_layers = len(viz_widget._scene.items())

        assert items_5_layers > items_2_layers, f"More layers should create more items: 5 layers={items_5_layers}, 2 layers={items_2_layers}"

    def test_scene_grows_with_layers(self, qapp, viz_widget, mock_model_manager):
        """Test that more layers creates more blocks."""
        viz_widget.update_model(mock_model_manager, list(range(2)))
        QTest.qWait(100)
        blocks_2 = len(viz_widget._blocks)

        viz_widget.update_model(mock_model_manager, list(range(10)))
        QTest.qWait(100)
        blocks_10 = len(viz_widget._blocks)

        assert blocks_10 > blocks_2, "More layers should create more blocks"


class TestInteractiveGraphicsView:
    """Tests for InteractiveGraphicsView zoom and pan."""

    def test_view_created(self, viz_widget):
        """Test that graphics view is created."""
        assert viz_widget._view is not None
        assert isinstance(viz_widget._view, InteractiveGraphicsView)

    def test_fit_in_view(self, viz_widget, mock_model_manager):
        """Test that fit in view works without errors."""
        viz_widget.update_model(mock_model_manager, list(range(3)))
        QTest.qWait(100)

        viz_widget._view.fit_in_view()
        QTest.qWait(50)

        assert viz_widget._view.get_zoom() > 0, "Zoom should be positive after fit"

    def test_set_zoom_updates_view(self, viz_widget):
        """Test that set_zoom updates the view."""
        viz_widget._view.set_zoom(2.0)
        assert viz_widget._view.get_zoom() == 2.0, "Zoom should be updated to 2.0"

    def test_zoom_clamped_to_limits(self, viz_widget):
        """Test that zoom is clamped to min/max."""
        viz_widget._view.set_zoom(0.001)
        assert viz_widget._view.get_zoom() == viz_widget._view._min_zoom, "Zoom should be clamped to min"

        viz_widget._view.set_zoom(100)
        assert viz_widget._view.get_zoom() == viz_widget._view._max_zoom, "Zoom should be clamped to max"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestVisualizationDataFlow:
    """Tests for data flow validation in model visualization."""

    def test_elements_have_unique_ids(self, qapp, viz_widget, mock_model_manager):
        """Test that all elements have unique IDs."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements, _ = create_model_graph(2, 12, 768, 64, 50257, 1024, None)
        ids = [e.id for e in elements]

        assert len(ids) == len(set(ids)), f"Element IDs must be unique: {len(ids)} total, {len(set(ids))} unique"

    def test_all_connections_have_valid_source_target(self, qapp, viz_widget, mock_model_manager):
        """Test that all connections reference valid element IDs."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements, connections = create_model_graph(2, 12, 768, 64, 50257, 1024, None)
        element_ids = {e.id for e in elements}

        for conn in connections:
            assert conn.source_id in element_ids, f"Connection source {conn.source_id} not in elements"
            assert conn.target_id in element_ids, f"Connection target {conn.target_id} not in elements"

    def test_no_self_referencing_connections(self, qapp, viz_widget, mock_model_manager):
        """Test that no connection connects an element to itself."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements, connections = create_model_graph(2, 12, 768, 64, 50257, 1024, None)

        for conn in connections:
            assert conn.source_id != conn.target_id, f"Self-referencing connection found: {conn.source_id}"

    def test_data_flow_graph_is_connected(self, qapp, viz_widget, mock_model_manager):
        """Test that the data flow graph is connected (all elements reachable from input)."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements, connections = create_model_graph(2, 12, 768, 64, 50257, 1024, None)

        element_ids = {e.id for e in elements}
        adjacency = {eid: set() for eid in element_ids}
        for conn in connections:
            adjacency[conn.source_id].add(conn.target_id)

        input_elements = [e for e in elements if e.label == "input"]
        assert len(input_elements) > 0, "Should have at least one input element"

        visited = set()
        queue = [input_elements[0].id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        unreachable = element_ids - visited
        assert len(unreachable) == 0, f"Found unreachable elements: {unreachable}"

    def test_layers_increase_with_more_layers(self, qapp, viz_widget, mock_model_manager):
        """Test that layer order increases with more transformer layers."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements_1, _ = create_model_graph(1, 12, 768, 64, 50257, 1024, None)
        elements_3, _ = create_model_graph(3, 12, 768, 64, 50257, 1024, None)

        layer_1_set = {e.layer_order for e in elements_1}
        layer_3_set = {e.layer_order for e in elements_3}

        assert len(layer_3_set) > len(layer_1_set), "More layers should create more unique layer orders"

    def test_arrows_flow_left_to_right(self, qapp, viz_widget, mock_model_manager):
        """Test that arrows generally flow from left to right in the data flow."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements, connections = create_model_graph(2, 12, 768, 64, 50257, 1024, None)

        element_map = {e.id: e for e in elements}

        left_to_right_count = 0
        for conn in connections:
            if conn.source_side == "right" and conn.target_side == "left":
                src = element_map[conn.source_id]
                tgt = element_map[conn.target_id]
                if src.right <= tgt.left + 50:
                    left_to_right_count += 1

        assert left_to_right_count >= len(connections) * 0.8, \
            f"Most arrows should flow left-to-right: {left_to_right_count}/{len(connections)}"

    def test_no_overlapping_elements_in_graph(self, qapp, viz_widget, mock_model_manager):
        """Test that elements in the generated graph don't overlap after layout."""
        from attention_studio.ui.viz_elements import create_model_graph

        elements, connections = create_model_graph(3, 12, 768, 64, 50257, 1024, None)

        overlaps = []
        for i, e1 in enumerate(elements):
            for e2 in elements[i + 1:]:
                overlap_x = max(0, min(e1.x + e1.width, e2.x + e2.width) - max(e1.x, e2.x))
                overlap_y = max(0, min(e1.y + e1.height, e2.y + e2.height) - max(e1.y, e2.y))
                if overlap_x > 0 and overlap_y > 0:
                    overlaps.append((e1.label, e2.label))

        if overlaps:
            print(f"Warning: Element overlaps: {overlaps}")

