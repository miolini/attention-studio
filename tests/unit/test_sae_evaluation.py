import pytest
import torch
import torch.nn as nn
from attention_studio.core.sae_evaluation import (
    SAEEvaluationSuite,
    SAEComparison,
    SAEAnalyzer,
    SAEEvaluationResult,
)


class MockTranscoder(nn.Module):
    def __init__(self, input_dim=512, dictionary_size=1024):
        super().__init__()
        self.decoder = nn.Linear(dictionary_size, input_dim)
        self.encoder = nn.Linear(input_dim, dictionary_size)
        self.config = type('obj', (object,), {'dictionary_size': dictionary_size})()

    def forward(self, x):
        z = self.encoder(x)
        features = torch.relu(z)
        reconstructed = self.decoder(features)
        return reconstructed, features


@pytest.fixture
def mock_transcoder():
    return MockTranscoder(input_dim=128, dictionary_size=256)


@pytest.fixture
def hidden_states():
    return torch.randn(4, 10, 128)


class TestSAEEvaluationSuite:
    def test_reconstruction_error(self, mock_transcoder, hidden_states):
        suite = SAEEvaluationSuite(mock_transcoder)
        error = suite.compute_reconstruction_error(hidden_states)
        assert isinstance(error, float)
        assert error >= 0

    def test_feature_sparsity(self, mock_transcoder, hidden_states):
        suite = SAEEvaluationSuite(mock_transcoder)
        sparsity = suite.compute_feature_sparsity(hidden_states)
        assert "sparsity_ratio" in sparsity
        assert 0 <= sparsity["sparsity_ratio"] <= 1

    def test_feature_utilization(self, mock_transcoder, hidden_states):
        suite = SAEEvaluationSuite(mock_transcoder)
        util = suite.compute_feature_utilization(hidden_states)
        assert "utilization" in util
        assert "dead_features" in util

    def test_explained_variance(self, mock_transcoder, hidden_states):
        suite = SAEEvaluationSuite(mock_transcoder)
        var = suite.compute_explained_variance(hidden_states)
        assert -1 <= var <= 1

    def test_ceiling_analysis(self, mock_transcoder, hidden_states):
        suite = SAEEvaluationSuite(mock_transcoder)
        results = suite.compute_ceiling_analysis(hidden_states, [10, 50, 100])
        assert 10 in results
        assert 50 in results

    def test_full_evaluation(self, mock_transcoder, hidden_states):
        suite = SAEEvaluationSuite(mock_transcoder)
        result = suite.evaluate(hidden_states)
        assert isinstance(result, SAEEvaluationResult)
        assert result.reconstruction_error >= 0


class TestSAEComparison:
    def test_compare_reconstruction(self):
        r1 = SAEEvaluationResult(
            reconstruction_error=0.1,
            sparsity=0.5,
            l0_norm=10.0,
            l1_norm=5.0,
            feature_utilization=0.8,
            dead_features=0.2,
            explained_variance=0.9,
        )
        r2 = SAEEvaluationResult(
            reconstruction_error=0.05,
            sparsity=0.6,
            l0_norm=8.0,
            l1_norm=4.0,
            feature_utilization=0.9,
            dead_features=0.1,
            explained_variance=0.95,
        )
        comparison = SAEComparison.compare_reconstruction(r1, r2)
        assert "reconstruction_improvement" in comparison
        assert "sparsity_improvement" in comparison

    def test_rank_saes(self):
        results = [
            SAEEvaluationResult(0.1, 0.5, 10, 5, 0.8, 0.2, 0.9),
            SAEEvaluationResult(0.05, 0.6, 8, 4, 0.9, 0.1, 0.95),
            SAEEvaluationResult(0.15, 0.4, 12, 6, 0.7, 0.3, 0.85),
        ]
        ranks = SAEComparison.rank_saes(results)
        assert len(ranks) == 3


class TestSAEAnalyzer:
    def test_analyze_feature_neurons(self, mock_transcoder, hidden_states):
        analyzer = SAEAnalyzer(mock_transcoder)
        results = analyzer.analyze_feature_neurons(hidden_states, top_k=10)
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_detect_anomalies(self, mock_transcoder, hidden_states):
        analyzer = SAEAnalyzer(mock_transcoder)
        result = analyzer.detect_anomalies(hidden_states)
        assert "anomaly_indices" in result
        assert "anomaly_count" in result

    def test_compute_feature_diversity(self, mock_transcoder, hidden_states):
        analyzer = SAEAnalyzer(mock_transcoder)
        result = analyzer.compute_feature_diversity(hidden_states)
        assert "diversity_ratio" in result
        assert 0 <= result["diversity_ratio"] <= 1


class TestEdgeCases:
    def test_empty_hidden_states(self, mock_transcoder):
        suite = SAEEvaluationSuite(mock_transcoder)
        hidden_states = torch.randn(0, 10, 128)
        result = suite.evaluate(hidden_states)
        assert isinstance(result, SAEEvaluationResult)

    def test_single_sample(self, mock_transcoder):
        suite = SAEEvaluationSuite(mock_transcoder)
        hidden_states = torch.randn(1, 5, 128)
        result = suite.evaluate(hidden_states)
        assert isinstance(result, SAEEvaluationResult)
