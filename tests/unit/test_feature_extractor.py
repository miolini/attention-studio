import pytest
import torch

from attention_studio.core.crm import Lorsa, LorsaConfig, Transcoder, TranscoderConfig
from attention_studio.core.feature_extractor import (
    FeatureExtractor,
    GlobalCircuit,
    GlobalCircuitAnalyzer,
)


class MockModelManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def is_loaded(self):
        return True


class TestFeatureExtractor:
    def test_feature_extractor_initialization(self):
        config = TranscoderConfig(dictionary_size=1024, top_k=64)
        transcoder = Transcoder(input_dim=768, config=config)

        class MockTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.randint(0, 50000, (1, 10))}

        class MockModel:
            device = torch.device("cpu")

        model_manager = MockModelManager(MockModel(), MockTokenizer())

        extractor = FeatureExtractor(model_manager, transcoder, layer_idx=0)

        assert extractor.layer_idx == 0
        assert extractor.transcoder is transcoder
        assert extractor.model_manager is model_manager


class TestGlobalCircuitAnalyzer:
    def test_global_circuit_structure(self):
        circuit = GlobalCircuit(
            name="Test Circuit",
            circuit_type="induction",
            features=[(0, 1), (0, 2), (1, 5)],
            strength=0.75,
        )

        assert circuit.name == "Test Circuit"
        assert circuit.circuit_type == "induction"
        assert len(circuit.features) == 3
        assert circuit.strength == 0.75

    def test_global_circuit_analyzer_initialization(self):
        config = TranscoderConfig(dictionary_size=512, top_k=32)
        transcoder = Transcoder(input_dim=256, config=config)

        class MockTokenizer:
            pass

        class MockModel:
            device = torch.device("cpu")

        model_manager = MockModelManager(MockModel(), MockTokenizer())

        analyzer = GlobalCircuitAnalyzer(
            model_manager,
            [transcoder],
            None,
            [0],
        )

        assert analyzer.model_manager is model_manager
        assert len(analyzer.transcoders) == 1
        assert analyzer.layer_indices == [0]
        assert analyzer.lorsas is None

    def test_global_circuit_analyzer_with_lorsas(self):
        tc_config = TranscoderConfig(dictionary_size=512, top_k=32)
        transcoder = Transcoder(input_dim=256, config=tc_config)

        lorsa_config = LorsaConfig(num_heads=8, top_k=32)
        lorsa = Lorsa(hidden_size=256, config=lorsa_config)

        class MockTokenizer:
            pass

        class MockModel:
            device = torch.device("cpu")

        model_manager = MockModelManager(MockModel(), MockTokenizer())

        analyzer = GlobalCircuitAnalyzer(
            model_manager,
            [transcoder],
            [lorsa],
            [0],
        )

        assert analyzer.lorsas is not None
        assert len(analyzer.lorsas) == 1

    def test_compute_feature_circuits(self):
        tc_config = TranscoderConfig(dictionary_size=256, top_k=16)
        transcoder = Transcoder(input_dim=128, config=tc_config)

        class MockTokenizer:
            pass

        class MockModel:
            device = torch.device("cpu")

        model_manager = MockModelManager(MockModel(), MockTokenizer())

        analyzer = GlobalCircuitAnalyzer(
            model_manager,
            [transcoder],
            None,
            [0],
        )

        result = analyzer.compute_feature_circuits(layer_idx=0, feature_idx=5)

        assert result["layer_idx"] == 0
        assert result["feature_idx"] == 5
        assert "decoder_vec" in result
        assert "encoder_vec" in result
        assert "norm" in result
        assert result["qk_circuit"] is None
        assert result["ov_circuit"] is None

    def test_compute_feature_circuits_with_lorsa(self):
        tc_config = TranscoderConfig(dictionary_size=256, top_k=16)
        transcoder = Transcoder(input_dim=128, config=tc_config)

        lorsa_config = LorsaConfig(num_heads=4, top_k=16)
        lorsa = Lorsa(hidden_size=128, config=lorsa_config)

        class MockTokenizer:
            pass

        class MockModel:
            device = torch.device("cpu")

        model_manager = MockModelManager(MockModel(), MockTokenizer())

        analyzer = GlobalCircuitAnalyzer(
            model_manager,
            [transcoder],
            [lorsa],
            [0],
        )

        result = analyzer.compute_feature_circuits(layer_idx=0, feature_idx=5)

        assert result["qk_circuit"] is not None
        assert result["ov_circuit"] is not None
        assert "W_Q" in result["qk_circuit"]
        assert "W_K" in result["qk_circuit"]

    def test_compute_feature_circuits_invalid_layer(self):
        tc_config = TranscoderConfig(dictionary_size=256, top_k=16)
        transcoder = Transcoder(input_dim=128, config=tc_config)

        class MockTokenizer:
            pass

        class MockModel:
            device = torch.device("cpu")

        model_manager = MockModelManager(MockModel(), MockTokenizer())

        analyzer = GlobalCircuitAnalyzer(
            model_manager,
            [transcoder],
            None,
            [0],
        )

        with pytest.raises(ValueError):
            analyzer.compute_feature_circuits(layer_idx=5, feature_idx=0)
