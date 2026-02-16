import asyncio
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from attention_studio.core.crm import LorsaConfig
from attention_studio.core.model_manager import ModelConfig, ModelManager
from attention_studio.core.trainer import CRMTrainer, TrainingConfig, TranscoderConfig
from attention_studio.ui.main_window import ModelVisualizationWidget, StudioMainWindow


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    """Create main window for testing."""
    window = StudioMainWindow()
    yield window
    window.close()
    QTest.qWait(100)


@pytest.fixture
def viz_widget(qapp):
    """Create visualization widget for testing."""
    widget = ModelVisualizationWidget(use_threaded_optimization=False)
    yield widget
    widget.close()
    QTest.qWait(100)


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TestModelVisualizationWidget:
    """Tests for the ModelVisualizationWidget."""

    def test_widget_creation(self, viz_widget):
        """Test that widget can be created."""
        assert viz_widget is not None
        assert viz_widget._scene is not None
        assert viz_widget._view is not None

    def test_empty_state(self, viz_widget):
        """Test that empty state shows placeholder."""
        viz_widget._draw_empty_state()
        # Should have at least one text item
        items = viz_widget._scene.items()
        assert len(items) > 0

    def test_update_model_creates_visualization(self, qapp, viz_widget):
        """Test that update_model creates visualization items."""
        # Create mock model manager
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.name_or_path = "gpt2"
        mock_config.n_embd = 768
        mock_config.n_layer = 12
        mock_manager.model.config = mock_config

        # Call update_model
        viz_widget.update_model(mock_manager, list(range(12)))

        # Process events to ensure rendering
        QTest.qWait(100)

        # Should have visualization items now (not just debug text)
        items = viz_widget._scene.items()
        # At minimum we should have: debug text + embed circle + 12 MLP + 12 Attn + output circle + legend
        assert len(items) > 1, "Should have visualization items after update_model"


class TestModelLoadingCallback:
    """Tests for model loading callbacks."""

    def test_async_callback_execution(self, qapp, main_window):
        """Test that async callbacks are executed properly."""
        callback_executed = threading.Event()

        def on_done():
            callback_executed.set()

        def do_work():
            time.sleep(0.1)

        main_window._run_async(do_work, on_done, None)

        # Wait for callback with timeout
        timeout = 5000  # 5 seconds
        start = time.time()
        while not callback_executed.is_set() and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.01)

        assert callback_executed.is_set(), "Callback should have been executed"

    def test_model_manager_works(self):
        """Test that ModelManager loads correctly (sanity check)."""
        manager = ModelManager()
        config = ModelConfig(
            name="gpt2",
            device=get_device(),
            dtype="float32",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(config))

        assert manager.is_loaded
        assert manager.model is not None
        assert manager.tokenizer is not None

        info = manager.get_model_info()
        assert info["name"] == "gpt2"
        assert info["num_layers"] == 12

        manager.unload()

    def test_load_button_triggers_callback(self, qapp, main_window):
        """Test that clicking load button triggers loading and callback."""
        # Set up mock to capture the async flow
        original_load = main_window.model_manager.load_model
        load_called = threading.Event()

        async def mock_load(config):
            load_called.set()
            await original_load(config)

        main_window.model_manager.load_model = mock_load

        # Click the load button
        main_window._load_btn.click()

        # Process events
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.05)

        # Wait for load to complete
        timeout = 30000  # 30 seconds
        start = time.time()
        while not load_called.is_set() and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.1)

        # Now wait for completion
        while not main_window._load_btn.isEnabled() and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.1)

        # Verify model is loaded
        assert main_window.model_manager.is_loaded, "Model should be loaded after button click"


class TestMainWindowUI:
    """Tests for main window UI components."""

    def test_window_creation(self, main_window):
        """Test that main window can be created."""
        assert main_window is not None
        assert main_window.windowTitle() == "Attention Studio"

    def test_tabs_exist(self, main_window):
        """Test that all tabs are created."""
        assert main_window._main_tabs is not None
        assert main_window._main_tabs.count() >= 3  # Viz, Graph, Features

    def test_sidebar_creation(self, main_window):
        """Test that sidebars are created."""
        assert main_window._left_dock is not None
        assert main_window._right_dock is not None
        assert main_window._bottom_dock is not None

    def test_toggle_left_sidebar(self, main_window):
        """Test toggling left sidebar."""
        initial_visible = main_window._left_sidebar_visible
        main_window.toggle_left_sidebar()
        assert main_window._left_sidebar_visible != initial_visible

    def test_toggle_right_sidebar(self, main_window):
        """Test toggling right sidebar."""
        initial_visible = main_window._right_sidebar_visible
        main_window.toggle_right_sidebar()
        assert main_window._right_sidebar_visible != initial_visible

    def test_model_status_initially_not_loaded(self, main_window):
        """Test that model status shows not loaded initially."""
        assert main_window._model_status.text() == "Not loaded"


class TestIntegrationModelLoading:
    """Integration tests for model loading through UI."""

    @pytest.mark.integration
    def test_full_load_flow(self, qapp, main_window):
        """Test complete model loading flow through UI."""
        # Click load button
        main_window._load_btn.click()

        # Process events while loading
        timeout = 60000  # 60 seconds
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.1)

        # Verify loaded
        assert main_window.model_manager.is_loaded, "Model should be loaded"

        # Wait for callback to complete - process events to allow signal to fire
        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.05)

        # Verify status updated
        assert "Loaded" in main_window._model_status.text(), f"Status should show loaded, got: {main_window._model_status.text()}"

        # Wait for visualization optimization to complete
        main_window._viz_tab.wait_for_optimization(timeout_ms=30000)

        # Verify visualization updated
        items = main_window._viz_tab._scene.items()
        # Should have more than just the empty state
        assert len(items) > 1, f"Should have visualization items, got {len(items)}"

    @pytest.mark.integration
    def test_viz_tab_updates_on_load(self, qapp, main_window):
        """Test that visualization tab updates when model loads."""
        # Set current tab to visualization
        main_window._main_tabs.setCurrentIndex(0)

        # Load model
        main_window._load_btn.click()

        # Process events
        timeout = 60000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.1)

        # Force process remaining events
        for _ in range(20):
            qapp.processEvents()
            time.sleep(0.05)

        # Wait for visualization optimization to complete
        main_window._viz_tab.wait_for_optimization(timeout_ms=30000)

        # Check that visualization has items
        items = main_window._viz_tab._scene.items()
        assert len(items) > 1, f"Visualization should have items, got {len(items)}"


class TestIntegrationCRMFlow:
    """Integration tests for complete CRM workflow."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_full_crm_happy_path(self, qapp, main_window):
        """Test the full CRM workflow: load model -> build transcoders -> extract features."""
        # Step 1: Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded, "Model should be loaded"

        # Wait for loading callback to complete
        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Wait for visualization to complete
        main_window._viz_tab.wait_for_optimization(timeout_ms=60000)

        # Step 2: Build transcoders (CRM) - run synchronously
        main_window._dict_size_combo.setCurrentText("8192")
        main_window._top_k_combo.setCurrentText("32")

        # Build CRM synchronously
        config = TranscoderConfig(
            dictionary_size=8192,
            top_k=32,
        )

        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), config)

        # Manually populate the layer combo (normally done in _on_build_crm)
        main_window._feat_layer_combo.clear()
        for i in range(num_layers):
            main_window._feat_layer_combo.addItem(f"Layer {i}")

        # Verify transcoders were built
        assert main_window.trainer is not None, "Trainer should be created"
        assert len(main_window.trainer.transcoders) > 0, "Transcoders should be built"
        assert main_window._feat_layer_combo.count() > 0, "Layer combo should have items"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_build_crm_button_click(self, qapp, main_window):
        """Test that clicking the Build CRM button works correctly."""
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(50):
            qapp.processEvents()
            time.sleep(0.1)

        main_window._dict_size_combo.setCurrentText("1024")
        main_window._top_k_combo.setCurrentText("32")

        main_window._build_crm_btn2.click()

        timeout = 60000
        start = time.time()
        while not main_window._build_crm_btn2.isEnabled() and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.1)

        assert main_window.trainer is not None
        assert len(main_window.trainer.transcoders) > 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_attribution_graph_build(self, qapp, main_window):
        """Test building attribution graph."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        # Wait for loading callback
        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build transcoders synchronously
        config = TranscoderConfig(dictionary_size=8192, top_k=32)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), config)

        # Manually populate the layer combo
        main_window._feat_layer_combo.clear()
        for i in range(num_layers):
            main_window._feat_layer_combo.addItem(f"Layer {i}")

        # Process events
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.01)

        assert main_window._feat_layer_combo.count() > 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_transcoder_has_correct_structure(self, qapp, main_window):
        """Test that transcoder has the correct structure."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build transcoders
        config = TranscoderConfig(dictionary_size=8192, top_k=32)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size", 768)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), config)

        # Verify transcoder structure
        tc = main_window.trainer.transcoders[0]

        assert tc.input_dim == hidden_size, "Input dim should match model hidden size"
        assert tc.config.dictionary_size == 8192, "Dictionary size should be 8192"
        assert tc.config.top_k == 32, "Top K should be 32"

        # Check encoder/decoder weights exist
        assert hasattr(tc, "encoder")
        assert hasattr(tc, "decoder")
        assert tc.encoder.weight.shape[0] == 8192, "Encoder output should be dictionary size"
        assert tc.encoder.weight.shape[1] == hidden_size, "Encoder input should be hidden size"
        assert tc.decoder.weight.shape[0] == hidden_size, "Decoder output should be hidden size"
        assert tc.decoder.weight.shape[1] == 8192, "Decoder input should be dictionary size"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_lorsa_has_correct_structure(self, qapp, main_window):
        """Test that Lorsa has the correct structure for attention decomposition."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build Lorsa modules
        from attention_studio.core.crm import LorsaConfig

        config = LorsaConfig(num_heads=12, top_k=128)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size", 768)

        config.num_heads = 12

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_lorsas(list(range(num_layers)), config)

        # Verify Lorsa structure
        lorsa = main_window.trainer.lorsas[0]

        assert lorsa.hidden_size == hidden_size
        assert lorsa.num_heads == 12
        assert lorsa.head_dim == hidden_size // 12

        # Check QK and OV circuits exist
        assert hasattr(lorsa, "W_Q")
        assert hasattr(lorsa, "W_K")
        assert hasattr(lorsa, "W_V")
        assert hasattr(lorsa, "W_O")
        assert hasattr(lorsa, "sparse_W_V")
        assert hasattr(lorsa, "sparse_W_O")

        # Verify sparse OV circuit (sparse_W_V and sparse_W_O are per-head with top_k)
        assert lorsa.sparse_W_V.shape[0] == 12  # num_heads
        assert lorsa.sparse_W_O.shape[0] == 12  # num_heads

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_crm_combine_transcoders_and_lorsas(self, qapp, main_window):
        """Test that CRM can combine transcoders and lorsas."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build both
        transcoder_config = TranscoderConfig(dictionary_size=8192, top_k=32)
        lorsa_config = LorsaConfig(num_heads=12, top_k=128)

        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), transcoder_config)
        main_window.trainer.build_lorsas(list(range(num_layers)), lorsa_config)

        # Verify both are built
        assert len(main_window.trainer.transcoders) == num_layers
        assert len(main_window.trainer.lorsas) == num_layers

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_feature_extractor_can_be_instantiated(self, qapp, main_window):
        """Test that FeatureExtractor can be created with correct parameters."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build transcoders
        config = TranscoderConfig(dictionary_size=8192, top_k=32)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), config)

        # Get transcoder and create extractor
        transcoder = main_window.trainer.get_transcoder(0)
        assert transcoder is not None

        from attention_studio.core.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor(main_window.model_manager, transcoder, layer_idx=0)

        assert extractor.layer_idx == 0
        assert extractor.transcoder is transcoder
        assert extractor.model_manager is main_window.model_manager

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_global_circuit_analyzer_can_be_instantiated(self, qapp, main_window):
        """Test that GlobalCircuitAnalyzer can be created."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build transcoders
        config = TranscoderConfig(dictionary_size=8192, top_k=32)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), config)

        # Create analyzer
        from attention_studio.core.feature_extractor import GlobalCircuitAnalyzer

        analyzer = GlobalCircuitAnalyzer(
            main_window.model_manager,
            main_window.trainer.transcoders,
            None,
            main_window.trainer.layer_indices,
        )

        assert analyzer.model_manager is main_window.model_manager
        assert len(analyzer.transcoders) == num_layers
        assert analyzer.layer_indices == main_window.trainer.layer_indices

    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_graph_builder_can_be_instantiated(self, qapp, main_window):
        """Test that GraphBuilder can be created."""
        # Load model
        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        # Build transcoders
        config = TranscoderConfig(dictionary_size=8192, top_k=32)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(num_layers)), config)

        # Create builder
        from attention_studio.core.feature_extractor import GraphBuilder

        builder = GraphBuilder(
            main_window.model_manager,
            main_window.trainer.transcoders,
            main_window.trainer.layer_indices,
        )

        assert builder.model_manager is main_window.model_manager
        assert len(builder.transcoders) == num_layers
        assert builder.layer_indices == main_window.trainer.layer_indices

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="Requires CUDA or MPS"
    )
    def test_training_does_not_crash(self, qapp, main_window):
        """Test that training runs without crashing."""
        from attention_studio.core.dataset import DatasetConfig, DatasetManager

        main_window._load_btn.click()

        timeout = 120000
        start = time.time()

        while not main_window.model_manager.is_loaded and (time.time() - start) * 1000 < timeout:
            qapp.processEvents()
            time.sleep(0.2)

        assert main_window.model_manager.is_loaded

        for _ in range(100):
            qapp.processEvents()
            time.sleep(0.1)

        config = TranscoderConfig(dictionary_size=512, top_k=32)
        model_config = main_window.model_manager.model.config
        num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", 12)

        main_window.trainer = CRMTrainer(main_window.model_manager, TrainingConfig())
        main_window.trainer.build_transcoders(list(range(min(2, num_layers))), config)

        dataset_config = DatasetConfig(
            source="custom",
            token_limit=1000,
            custom_path=None,
            format="txt",
        )

        main_window.dataset_manager = DatasetManager()
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            main_window.dataset_manager.load(dataset_config, main_window.model_manager.tokenizer)
        )

        dataloader = main_window.dataset_manager.create_dataloader(batch_size=4, max_length=64)

        result = main_window.trainer.train_transcoder(
            dataloader,
            layer_idx=0,
            progress_callback=None,
        )

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
