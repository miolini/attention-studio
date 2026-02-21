import pytest
from attention_studio.core.layer_analysis import (
    LayerRepresentationAnalyzer,
    LayerTransitionAnalyzer,
    LayerComparison,
    LayerAnalysis,
)


class TestLayerAnalysisImports:
    def test_import_layer_representation_analyzer(self):
        assert LayerRepresentationAnalyzer is not None

    def test_import_layer_transition_analyzer(self):
        assert LayerTransitionAnalyzer is not None

    def test_import_layer_comparison(self):
        assert LayerComparison is not None

    def test_import_layer_analysis_dataclass(self):
        assert LayerAnalysis is not None


class TestLayerAnalysisClass:
    def test_layer_analysis_creation(self):
        analysis = LayerAnalysis(
            layer_idx=5,
            representation_norm=1.0,
            representation_variance=0.5,
            cosine_similarity=0.9,
        )
        assert analysis.layer_idx == 5
        assert analysis.representation_norm == 1.0


class TestLayerComparison:
    def test_compare_layers(self):
        analysis1 = {1: 0.5, 2: 0.6, 3: 0.7}
        analysis2 = {1: 0.4, 2: 0.7, 3: 0.6}
        result = LayerComparison.compare_layers(analysis1, analysis2)
        assert 1 in result
        assert result[1] == pytest.approx(0.1)

    def test_compute_diversity_score(self):
        layer_metrics = {
            1: {"norm": 1.0, "var": 0.5},
            2: {"norm": 2.0, "var": 0.6},
        }
        score = LayerComparison.compute_diversity_score(layer_metrics)
        assert score >= 0

    def test_find_optimal_layer_max(self):
        layer_metrics = {1: 0.5, 2: 0.8, 3: 0.6}
        optimal = LayerComparison.find_optimal_layer(layer_metrics, criterion="max")
        assert optimal == 2

    def test_find_optimal_layer_min(self):
        layer_metrics = {1: 0.5, 2: 0.8, 3: 0.6}
        optimal = LayerComparison.find_optimal_layer(layer_metrics, criterion="min")
        assert optimal == 1

    def test_find_optimal_layer_empty(self):
        optimal = LayerComparison.find_optimal_layer({}, criterion="max")
        assert optimal is None
