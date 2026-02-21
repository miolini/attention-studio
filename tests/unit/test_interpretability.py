import pytest
import numpy as np
import torch
from attention_studio.core.interpretability import (
    FaithfulnessMetric,
    CompletenessMetric,
    LocalizationMetric,
    StabilityMetric,
    ReconstructionMetric,
    FeatureQualityMetric,
    InterpretabilityEvaluator,
    InterpretabilityMetric,
)


class TestFaithfulnessMetric:
    def test_correlation_perfect_positive(self):
        importance = [1.0, 2.0, 3.0, 4.0]
        outputs = [1.0, 2.0, 3.0, 4.0]
        result = FaithfulnessMetric.compute_correlation(importance, outputs)
        assert abs(result.value - 1.0) < 0.001

    def test_correlation_perfect_negative(self):
        importance = [1.0, 2.0, 3.0, 4.0]
        outputs = [4.0, 3.0, 2.0, 1.0]
        result = FaithfulnessMetric.compute_correlation(importance, outputs)
        assert abs(result.value + 1.0) < 0.001

    def test_correlation_no_relationship(self):
        importance = [1.0, 2.0, 3.0, 4.0]
        outputs = [2.0, 2.0, 2.0, 2.0]
        result = FaithfulnessMetric.compute_correlation(importance, outputs)
        assert result.value == 0.0

    def test_correlation_empty(self):
        result = FaithfulnessMetric.compute_correlation([], [])
        assert result.value == 0.0

    def test_sufficiency(self):
        importance = [1.0, 0.0, 0.0, 0.5]
        top_k = [0.9, 0.1]
        full = [1.0, 0.2]
        result = FaithfulnessMetric.compute_sufficiency(importance, top_k, full)
        assert 0.0 <= result.value <= 1.0


class TestCompletenessMetric:
    def test_attribution_sum_perfect(self):
        importance = [0.5, 0.3, 0.2]
        original = 1.0
        result = CompletenessMetric.compute_attribution_sum(importance, original)
        assert result.value > 0.9

    def test_attribution_sum_zero_original(self):
        importance = [0.5, 0.3, 0.2]
        original = 0.0
        result = CompletenessMetric.compute_attribution_sum(importance, original)
        assert result.value == 0.0


class TestLocalizationMetric:
    def test_sparsity_all_nonzero(self):
        importance = [1.0, 2.0, 3.0, 4.0]
        result = LocalizationMetric.compute_sparsity(importance)
        assert result.value == 0.0

    def test_sparsity_all_zero(self):
        importance = [0.0, 0.0, 0.0]
        result = LocalizationMetric.compute_sparsity(importance)
        assert result.value == 1.0

    def test_sparsity_mixed(self):
        importance = [1.0, 0.0, 0.0, 2.0]
        result = LocalizationMetric.compute_sparsity(importance)
        assert 0.0 < result.value < 1.0

    def test_concentration(self):
        importance = [10.0, 1.0, 1.0, 1.0]
        result = LocalizationMetric.compute_concentration(importance)
        assert result.value > 0.5

    def test_concentration_empty(self):
        result = LocalizationMetric.compute_concentration([])
        assert result.value == 0.0


class TestStabilityMetric:
    def test_sensitivity_identical(self):
        scores1 = [1.0, 2.0, 3.0, 4.0]
        scores2 = [1.0, 2.0, 3.0, 4.0]
        result = StabilityMetric.compute_sensitivity(scores1, scores2)
        assert result.value == 0.0

    def test_sensitivity_different(self):
        scores1 = [1.0, 2.0, 3.0, 4.0]
        scores2 = [4.0, 3.0, 2.0, 1.0]
        result = StabilityMetric.compute_sensitivity(scores1, scores2)
        assert result.value > 0.0

    def test_variance(self):
        scores_list = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9],
        ]
        result = StabilityMetric.compute_variance(scores_list)
        assert result.value >= 0.0


class TestReconstructionMetric:
    def test_mse_perfect(self):
        original = torch.tensor([1.0, 2.0, 3.0])
        reconstructed = torch.tensor([1.0, 2.0, 3.0])
        result = ReconstructionMetric.compute_mse(original, reconstructed)
        assert result.value == 0.0

    def test_mse_different(self):
        original = torch.tensor([1.0, 2.0, 3.0])
        reconstructed = torch.tensor([2.0, 3.0, 4.0])
        result = ReconstructionMetric.compute_mse(original, reconstructed)
        assert result.value > 0.0

    def test_cosine_similarity_perfect(self):
        original = torch.tensor([1.0, 2.0, 3.0])
        reconstructed = torch.tensor([1.0, 2.0, 3.0])
        result = ReconstructionMetric.compute_cosine_similarity(original, reconstructed)
        assert abs(result.value - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        original = torch.tensor([1.0, 0.0])
        reconstructed = torch.tensor([0.0, 1.0])
        result = ReconstructionMetric.compute_cosine_similarity(original, reconstructed)
        assert abs(result.value) < 0.001

    def test_variance_explained(self):
        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        reconstructed = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ReconstructionMetric.compute_variance_explained(original, reconstructed)
        assert result.value > 0.9


class TestFeatureQualityMetric:
    def test_l0_norm(self):
        activations = torch.tensor([[1.0, 0.0, 2.0, 0.0, 3.0]])
        result = FeatureQualityMetric.compute_l0_norm(activations)
        assert result.value == 0.6

    def test_feature_utilization(self):
        activations = torch.tensor([[1.0, 0.0, 2.0, 0.0, 3.0]])
        result = FeatureQualityMetric.compute_feature_utilization(activations)
        assert result.value == 0.6

    def test_dead_features(self):
        activations = torch.tensor([[1.0, 0.0, 0.0, 2.0, 0.0]])
        result = FeatureQualityMetric.compute_dead_features(activations)
        assert result.value == 0.6

    def test_dead_features_all_alive(self):
        activations = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = FeatureQualityMetric.compute_dead_features(activations)
        assert result.value == 0.0


class TestInterpretabilityEvaluator:
    def test_evaluate_faithfulness(self):
        evaluator = InterpretabilityEvaluator()
        importance = [1.0, 2.0, 3.0, 4.0]
        outputs = [1.0, 2.0, 3.0, 4.0]
        results = evaluator.evaluate_faithfulness(importance, outputs)
        assert "correlation" in results

    def test_evaluate_completeness(self):
        evaluator = InterpretabilityEvaluator()
        importance = [0.5, 0.3, 0.2]
        original = 1.0
        results = evaluator.evaluate_completeness(importance, original)
        assert "attribution_sum" in results

    def test_evaluate_localization(self):
        evaluator = InterpretabilityEvaluator()
        importance = [1.0, 0.0, 0.0, 2.0]
        results = evaluator.evaluate_localization(importance)
        assert "sparsity" in results
        assert "concentration" in results

    def test_evaluate_stability(self):
        evaluator = InterpretabilityEvaluator()
        scores1 = [1.0, 2.0, 3.0]
        scores2 = [1.1, 2.1, 3.1]
        results = evaluator.evaluate_stability(scores1, scores2)
        assert "sensitivity" in results

    def test_evaluate_reconstruction(self):
        evaluator = InterpretabilityEvaluator()
        original = torch.tensor([1.0, 2.0, 3.0])
        reconstructed = torch.tensor([1.0, 2.0, 3.0])
        results = evaluator.evaluate_reconstruction(original, reconstructed)
        assert "mse" in results
        assert "cosine" in results
        assert "variance_explained" in results

    def test_evaluate_feature_quality(self):
        evaluator = InterpretabilityEvaluator()
        activations = torch.tensor([[1.0, 0.0, 2.0, 0.0]])
        results = evaluator.evaluate_feature_quality(activations)
        assert "l0_norm" in results
        assert "utilization" in results
        assert "dead_features" in results

    def test_get_summary(self):
        evaluator = InterpretabilityEvaluator()
        original = torch.tensor([1.0, 2.0, 3.0])
        reconstructed = torch.tensor([1.0, 2.0, 3.0])
        evaluator.evaluate_reconstruction(original, reconstructed)
        summary = evaluator.get_summary()
        assert isinstance(summary, dict)
