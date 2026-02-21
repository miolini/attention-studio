import pytest
from attention_studio.utils.metrics import (
    FeatureMetrics,
    GraphMetrics,
    CircuitMetrics,
    AttentionMetrics,
    ComparisonMetrics,
    Aggregator,
    MetricResult,
)


class TestFeatureMetrics:
    def test_sparsity(self):
        activations = [1.0, 0.0, 2.0, 0.0, 3.0]
        result = FeatureMetrics.compute_sparsity(activations)
        assert result.value == 40.0

    def test_l2_norm(self):
        activations = [3.0, 4.0]
        result = FeatureMetrics.compute_l2_norm(activations)
        assert result.value == 5.0

    def test_l1_norm(self):
        activations = [1.0, 2.0, 3.0]
        result = FeatureMetrics.compute_l1_norm(activations)
        assert result.value == 6.0

    def test_max(self):
        activations = [1.0, 5.0, 3.0]
        result = FeatureMetrics.compute_max(activations)
        assert result.value == 5.0

    def test_mean(self):
        activations = [1.0, 2.0, 3.0]
        result = FeatureMetrics.compute_mean(activations)
        assert result.value == 2.0

    def test_std(self):
        activations = [1.0, 2.0, 3.0]
        result = FeatureMetrics.compute_std(activations)
        assert abs(result.value - 0.816) < 0.01

    def test_percentiles(self):
        activations = list(range(100))
        results = FeatureMetrics.compute_percentiles(activations, [25, 50, 75])
        assert len(results) == 3
        assert results[0].value == 24
        assert results[1].value == 49
        assert results[2].value == 74

    def test_compute_all(self):
        activations = [1.0, 2.0, 3.0, 0.0, 5.0]
        results = FeatureMetrics.compute_all(activations)
        assert "sparsity" in results
        assert "l2_norm" in results
        assert "mean" in results

    def test_empty_activations(self):
        result = FeatureMetrics.compute_sparsity([])
        assert result.value == 0.0


class TestGraphMetrics:
    def test_density(self):
        result = GraphMetrics.compute_density(5, 10)
        expected = 10 / (5 * 4)
        assert abs(result.value - expected) < 0.001

    def test_avg_degree(self):
        result = GraphMetrics.compute_avg_degree(10, 5)
        assert result.value == 4.0

    def test_connectivity(self):
        edges = [("a", "b"), ("b", "c")]
        result = GraphMetrics.compute_connectivity(edges, 4)
        assert result.value == 0.75

    def test_avg_weight(self):
        edges = [{"weight": 1.0}, {"weight": 2.0}, {"weight": 3.0}]
        result = GraphMetrics.compute_avg_weight(edges)
        assert result.value == 2.0


class TestCircuitMetrics:
    def test_circuit_strength(self):
        features = [(0, 1), (1, 2), (2, 3)]
        result = CircuitMetrics.compute_circuit_strength(features)
        assert result.value == 3

    def test_feature_overlap(self):
        circuit1 = [(0, 1), (1, 2), (2, 3)]
        circuit2 = [(1, 2), (3, 4), (5, 6)]
        result = CircuitMetrics.compute_feature_overlap(circuit1, circuit2)
        assert result.value > 0

    def test_layer_spread(self):
        features = [(0, 1), (1, 2), (2, 3), (0, 4)]
        result = CircuitMetrics.compute_layer_spread(features)
        assert result.value == 3


class TestAttentionMetrics:
    def test_attention_entropy(self):
        attention = [[0.5, 0.5], [0.5, 0.5]]
        result = AttentionMetrics.compute_attention_entropy(attention)
        assert result.value == 1.0

    def test_attention_diversity(self):
        attention = [[0.9, 0.1], [0.8, 0.2]]
        result = AttentionMetrics.compute_attention_diversity(attention)
        assert result.value > 0

    def test_attention_concentration(self):
        attention = [[0.9, 0.1], [0.8, 0.2]]
        result = AttentionMetrics.compute_attention_concentration(attention)
        assert result.value > 50


class TestComparisonMetrics:
    def test_cosine_similarity(self):
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0]
        result = ComparisonMetrics.compute_cosine_similarity(vec1, vec2)
        assert result.value == 1.0

    def test_cosine_similarity_orthogonal(self):
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = ComparisonMetrics.compute_cosine_similarity(vec1, vec2)
        assert result.value == 0.0

    def test_euclidean_distance(self):
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        result = ComparisonMetrics.compute_euclidean_distance(vec1, vec2)
        assert result.value == 5.0

    def test_kl_divergence(self):
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        result = ComparisonMetrics.compute_kl_divergence(p, q)
        assert result.value == 0.0


class TestAggregator:
    def test_add_metric(self):
        aggregator = Aggregator()
        aggregator.add(MetricResult(name="loss", value=0.5))
        aggregator.add(MetricResult(name="loss", value=0.3))
        results = aggregator.summarize()
        assert "loss_mean" in results

    def test_add_dict(self):
        aggregator = Aggregator()
        metrics = {
            "accuracy": MetricResult(name="accuracy", value=0.9),
            "loss": MetricResult(name="loss", value=0.1),
        }
        aggregator.add_dict(metrics)
        results = aggregator.summarize()
        assert "accuracy_mean" in results
        assert "loss_mean" in results

    def test_summarize(self):
        aggregator = Aggregator()
        aggregator.add(MetricResult(name="val", value=1.0))
        aggregator.add(MetricResult(name="val", value=2.0))
        aggregator.add(MetricResult(name="val", value=3.0))
        results = aggregator.summarize()
        assert results["val_mean"].value == 2.0
        assert results["val_min"].value == 1.0
        assert results["val_max"].value == 3.0
