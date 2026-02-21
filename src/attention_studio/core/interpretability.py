from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class InterpretabilityMetric:
    name: str
    value: float
    description: str


class FaithfulnessMetric:
    @staticmethod
    def compute_correlation(
        importance_scores: list[float],
        model_outputs: list[float],
    ) -> InterpretabilityMetric:
        if len(importance_scores) != len(model_outputs) or len(importance_scores) < 2:
            return InterpretabilityMetric(
                name="faithfulness_correlation",
                value=0.0,
                description="Correlation between feature importance and model output change",
            )

        mean_imp = sum(importance_scores) / len(importance_scores)
        mean_out = sum(model_outputs) / len(model_outputs)

        numerator = sum(
            (imp - mean_imp) * (out - mean_out)
            for imp, out in zip(importance_scores, model_outputs, strict=True)
        )
        denom_imp = math.sqrt(sum((imp - mean_imp) ** 2 for imp in importance_scores))
        denom_out = math.sqrt(sum((out - mean_out) ** 2 for out in model_outputs))

        if denom_imp == 0 or denom_out == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denom_imp * denom_out)

        return InterpretabilityMetric(
            name="faithfulness_correlation",
            value=correlation,
            description="Correlation between feature importance and model output change",
        )

    @staticmethod
    def compute_sufficiency(
        importance_scores: list[float],
        model_outputs_top_k: list[float],
        model_outputs_full: list[float],
    ) -> InterpretabilityMetric:
        if len(model_outputs_top_k) != len(model_outputs_full):
            return InterpretabilityMetric(
                name="faithfulness_sufficiency",
                value=0.0,
                description="How well top-k features preserve model behavior",
            )

        if len(importance_scores) == 0:
            return InterpretabilityMetric(
                name="faithfulness_sufficiency",
                value=0.0,
                description="How well top-k features preserve model behavior",
            )

        top_k_ratio = len([s for s in importance_scores if s > 0]) / len(importance_scores)

        diffs = [
            abs(top_k - full)
            for top_k, full in zip(model_outputs_top_k, model_outputs_full, strict=True)
        ]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0

        sufficiency = 1.0 - min(avg_diff, 1.0)

        return InterpretabilityMetric(
            name="faithfulness_sufficiency",
            value=sufficiency * top_k_ratio,
            description="How well top-k features preserve model behavior",
        )


class CompletenessMetric:
    @staticmethod
    def compute_attribution_sum(
        importance_scores: list[float],
        original_output: float,
    ) -> InterpretabilityMetric:
        if not importance_scores or original_output == 0:
            return InterpretabilityMetric(
                name="completeness_attribution_sum",
                value=0.0,
                description="How well importance scores sum to original output",
            )

        attr_sum = sum(importance_scores)
        ratio = attr_sum / abs(original_output) if original_output != 0 else 0.0

        completeness = 1.0 - min(abs(1.0 - ratio), 1.0)

        return InterpretabilityMetric(
            name="completeness_attribution_sum",
            value=completeness,
            description="How well importance scores sum to original output",
        )


class LocalizationMetric:
    @staticmethod
    def compute_sparsity(importance_scores: list[float]) -> InterpretabilityMetric:
        if not importance_scores:
            return InterpretabilityMetric(
                name="localization_sparsity",
                value=0.0,
                description="How focused the importance is on few features",
            )

        nonzero = sum(1 for s in importance_scores if abs(s) > 1e-6)
        sparsity = 1.0 - (nonzero / len(importance_scores))

        return InterpretabilityMetric(
            name="localization_sparsity",
            value=sparsity,
            description="How focused the importance is on few features",
        )

    @staticmethod
    def compute_concentration(importance_scores: list[float]) -> InterpretabilityMetric:
        if not importance_scores:
            return InterpretabilityMetric(
                name="localization_concentration",
                value=0.0,
                description="How much importance is concentrated in top features",
            )

        sorted_scores = sorted([abs(s) for s in importance_scores], reverse=True)
        total = sum(sorted_scores)

        if total == 0:
            return InterpretabilityMetric(
                name="localization_concentration",
                value=0.0,
                description="How much importance is concentrated in top features",
            )

        cumsum = 0.0
        top_10_pct = max(1, len(sorted_scores) // 10)
        for i in range(top_10_pct):
            cumsum += sorted_scores[i]

        concentration = cumsum / total

        return InterpretabilityMetric(
            name="localization_concentration",
            value=concentration,
            description="How much importance is concentrated in top features",
        )


class StabilityMetric:
    @staticmethod
    def compute_sensitivity(
        importance_scores1: list[float],
        importance_scores2: list[float],
    ) -> InterpretabilityMetric:
        if len(importance_scores1) != len(importance_scores2):
            return InterpretabilityMetric(
                name="stability_sensitivity",
                value=1.0,
                description="How much explanations change with small input changes",
            )

        if not importance_scores1:
            return InterpretabilityMetric(
                name="stability_sensitivity",
                value=0.0,
                description="How much explanations change with small input changes",
            )

        diffs = [
            abs(s1 - s2)
            for s1, s2 in zip(importance_scores1, importance_scores2, strict=True)
        ]
        avg_diff = sum(diffs) / len(diffs)

        sensitivity = min(avg_diff, 1.0)

        return InterpretabilityMetric(
            name="stability_sensitivity",
            value=sensitivity,
            description="How much explanations change with small input changes",
        )

    @staticmethod
    def compute_variance(importance_scores_list: list[list[float]]) -> InterpretabilityMetric:
        if not importance_scores_list or len(importance_scores_list) < 2:
            return InterpretabilityMetric(
                name="stability_variance",
                value=0.0,
                description="Variance of importance scores across multiple runs",
            )

        num_features = len(importance_scores_list[0])
        if num_features == 0:
            return InterpretabilityMetric(
                name="stability_variance",
                value=0.0,
                description="Variance of importance scores across multiple runs",
            )

        variances = []
        for i in range(num_features):
            values = [scores[i] for scores in importance_scores_list if i < len(scores)]
            if values:
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / len(values)
                variances.append(var)

        avg_variance = sum(variances) / len(variances) if variances else 0.0

        return InterpretabilityMetric(
            name="stability_variance",
            value=avg_variance,
            description="Variance of importance scores across multiple runs",
        )


class ReconstructionMetric:
    @staticmethod
    def compute_mse(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> InterpretabilityMetric:
        if original.shape != reconstructed.shape:
            return InterpretabilityMetric(
                name="reconstruction_mse",
                value=float("inf"),
                description="Mean squared error between original and reconstructed",
            )

        mse = torch.mean((original - reconstructed) ** 2).item()

        return InterpretabilityMetric(
            name="reconstruction_mse",
            value=mse,
            description="Mean squared error between original and reconstructed",
        )

    @staticmethod
    def compute_cosine_similarity(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> InterpretabilityMetric:
        if original.shape != reconstructed.shape:
            return InterpretabilityMetric(
                name="reconstruction_cosine",
                value=0.0,
                description="Cosine similarity between original and reconstructed",
            )

        original_flat = original.flatten()
        reconstructed_flat = reconstructed.flatten()

        dot = torch.dot(original_flat, reconstructed_flat)
        norm_orig = torch.norm(original_flat)
        norm_recon = torch.norm(reconstructed_flat)

        if norm_orig == 0 or norm_recon == 0:
            similarity = 0.0
        else:
            similarity = (dot / (norm_orig * norm_recon)).item()

        return InterpretabilityMetric(
            name="reconstruction_cosine",
            value=similarity,
            description="Cosine similarity between original and reconstructed",
        )

    @staticmethod
    def compute_variance_explained(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> InterpretabilityMetric:
        if original.shape != reconstructed.shape:
            return InterpretabilityMetric(
                name="reconstruction_variance_explained",
                value=0.0,
                description="Variance of original explained by reconstruction",
            )

        original_flat = original.flatten()
        reconstructed_flat = reconstructed.flatten()

        var_original = torch.var(original_flat).item()
        var_residual = torch.var(original_flat - reconstructed_flat).item()

        if var_original == 0:
            variance_explained = 0.0
        else:
            variance_explained = 1.0 - (var_residual / var_original)

        return InterpretabilityMetric(
            name="reconstruction_variance_explained",
            value=variance_explained,
            description="Variance of original explained by reconstruction",
        )


class FeatureQualityMetric:
    @staticmethod
    def compute_l0_norm(activations: torch.Tensor) -> InterpretabilityMetric:
        l0 = (activations != 0).sum().item()
        total = activations.numel()
        normalized_l0 = l0 / total if total > 0 else 0.0

        return InterpretabilityMetric(
            name="feature_l0_norm",
            value=normalized_l0,
            description="Fraction of non-zero activations",
        )

    @staticmethod
    def compute_feature_utilization(
        feature_activations: torch.Tensor,
    ) -> InterpretabilityMetric:
        if feature_activations.numel() == 0:
            return InterpretabilityMetric(
                name="feature_utilization",
                value=0.0,
                description="Fraction of features with non-zero mean activation",
            )

        mean_activations = feature_activations.mean(dim=0)
        active_features = (mean_activations > 1e-6).sum().item()
        total_features = mean_activations.numel()

        utilization = active_features / total_features if total_features > 0 else 0.0

        return InterpretabilityMetric(
            name="feature_utilization",
            value=utilization,
            description="Fraction of features with non-zero mean activation",
        )

    @staticmethod
    def compute_dead_features(
        feature_activations: torch.Tensor,
        threshold: float = 1e-6,
    ) -> InterpretabilityMetric:
        if feature_activations.numel() == 0:
            return InterpretabilityMetric(
                name="feature_dead_features",
                value=1.0,
                description="Fraction of features that are never activated",
            )

        mean_activations = feature_activations.mean(dim=0)
        dead_features = (mean_activations.abs() < threshold).sum().item()
        total_features = mean_activations.numel()

        dead_ratio = dead_features / total_features if total_features > 0 else 1.0

        return InterpretabilityMetric(
            name="feature_dead_features",
            value=dead_ratio,
            description="Fraction of features that are never activated",
        )


class InterpretabilityEvaluator:
    def __init__(self):
        self.results: dict[str, InterpretabilityMetric] = {}

    def evaluate_faithfulness(
        self,
        importance_scores: list[float],
        model_outputs: list[float],
        model_outputs_top_k: Optional[list[float]] = None,
        model_outputs_full: Optional[list[float]] = None,
    ) -> dict[str, InterpretabilityMetric]:
        results = {}

        results["correlation"] = FaithfulnessMetric.compute_correlation(
            importance_scores, model_outputs
        )

        if model_outputs_top_k is not None and model_outputs_full is not None:
            results["sufficiency"] = FaithfulnessMetric.compute_sufficiency(
                importance_scores, model_outputs_top_k, model_outputs_full
            )

        self.results.update(results)
        return results

    def evaluate_completeness(
        self,
        importance_scores: list[float],
        original_output: float,
    ) -> dict[str, InterpretabilityMetric]:
        results = {}
        results["attribution_sum"] = CompletenessMetric.compute_attribution_sum(
            importance_scores, original_output
        )
        self.results.update(results)
        return results

    def evaluate_localization(
        self,
        importance_scores: list[float],
    ) -> dict[str, InterpretabilityMetric]:
        results = {}
        results["sparsity"] = LocalizationMetric.compute_sparsity(importance_scores)
        results["concentration"] = LocalizationMetric.compute_concentration(importance_scores)
        self.results.update(results)
        return results

    def evaluate_stability(
        self,
        importance_scores1: list[float],
        importance_scores2: list[float],
        importance_scores_list: Optional[list[list[float]]] = None,
    ) -> dict[str, InterpretabilityMetric]:
        results = {}
        results["sensitivity"] = StabilityMetric.compute_sensitivity(
            importance_scores1, importance_scores2
        )

        if importance_scores_list is not None:
            results["variance"] = StabilityMetric.compute_variance(importance_scores_list)

        self.results.update(results)
        return results

    def evaluate_reconstruction(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> dict[str, InterpretabilityMetric]:
        results = {}
        results["mse"] = ReconstructionMetric.compute_mse(original, reconstructed)
        results["cosine"] = ReconstructionMetric.compute_cosine_similarity(
            original, reconstructed
        )
        results["variance_explained"] = ReconstructionMetric.compute_variance_explained(
            original, reconstructed
        )
        self.results.update(results)
        return results

    def evaluate_feature_quality(
        self,
        feature_activations: torch.Tensor,
    ) -> dict[str, InterpretabilityMetric]:
        results = {}
        results["l0_norm"] = FeatureQualityMetric.compute_l0_norm(feature_activations)
        results["utilization"] = FeatureQualityMetric.compute_feature_utilization(
            feature_activations
        )
        results["dead_features"] = FeatureQualityMetric.compute_dead_features(
            feature_activations
        )
        self.results.update(results)
        return results

    def get_summary(self) -> dict[str, float]:
        return {name: metric.value for name, metric in self.results.items()}
