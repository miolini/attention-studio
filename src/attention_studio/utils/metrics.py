from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MetricResult:
    name: str
    value: float
    unit: Optional[str] = None
    metadata: Optional[dict] = None


class FeatureMetrics:
    @staticmethod
    def compute_sparsity(activations: list[float]) -> MetricResult:
        if not activations:
            return MetricResult(name="sparsity", value=0.0, unit="%")
        zero_count = sum(1 for a in activations if abs(a) < 1e-6)
        sparsity = (zero_count / len(activations)) * 100
        return MetricResult(name="sparsity", value=sparsity, unit="%")

    @staticmethod
    def compute_l2_norm(activations: list[float]) -> MetricResult:
        if not activations:
            return MetricResult(name="l2_norm", value=0.0)
        norm = math.sqrt(sum(a * a for a in activations))
        return MetricResult(name="l2_norm", value=norm)

    @staticmethod
    def compute_l1_norm(activations: list[float]) -> MetricResult:
        if not activations:
            return MetricResult(name="l1_norm", value=0.0)
        norm = sum(abs(a) for a in activations)
        return MetricResult(name="l1_norm", value=norm)

    @staticmethod
    def compute_max(activations: list[float]) -> MetricResult:
        if not activations:
            return MetricResult(name="max", value=0.0)
        return MetricResult(name="max", value=max(activations))

    @staticmethod
    def compute_mean(activations: list[float]) -> MetricResult:
        if not activations:
            return MetricResult(name="mean", value=0.0)
        return MetricResult(name="mean", value=sum(activations) / len(activations))

    @staticmethod
    def compute_std(activations: list[float]) -> MetricResult:
        if not activations:
            return MetricResult(name="std", value=0.0)
        mean = sum(activations) / len(activations)
        variance = sum((a - mean) ** 2 for a in activations) / len(activations)
        return MetricResult(name="std", value=math.sqrt(variance))

    @staticmethod
    def compute_percentiles(
        activations: list[float], percentiles: Optional[list[int]] = None
    ) -> list[MetricResult]:
        if not activations:
            return []
        percentiles = percentiles or [25, 50, 75, 90, 95, 99]
        sorted_activations = sorted(activations)
        n = len(sorted_activations)
        results = []
        for p in percentiles:
            idx = int((n - 1) * p / 100)
            idx = min(idx, n - 1)
            idx = max(idx, 0)
            results.append(MetricResult(name=f"p{p}", value=sorted_activations[idx]))
        return results

    @staticmethod
    def compute_all(activations: list[float]) -> dict[str, MetricResult]:
        metrics = {
            "sparsity": FeatureMetrics.compute_sparsity(activations),
            "l2_norm": FeatureMetrics.compute_l2_norm(activations),
            "l1_norm": FeatureMetrics.compute_l1_norm(activations),
            "max": FeatureMetrics.compute_max(activations),
            "mean": FeatureMetrics.compute_mean(activations),
            "std": FeatureMetrics.compute_std(activations),
        }
        metrics.update({m.name: m for m in FeatureMetrics.compute_percentiles(activations)})
        return metrics


class GraphMetrics:
    @staticmethod
    def compute_density(num_nodes: int, num_edges: int) -> MetricResult:
        if num_nodes <= 1:
            return MetricResult(name="density", value=0.0)
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0.0
        return MetricResult(name="density", value=density)

    @staticmethod
    def compute_avg_degree(num_edges: int, num_nodes: int) -> MetricResult:
        if num_nodes == 0:
            return MetricResult(name="avg_degree", value=0.0)
        avg_degree = (2 * num_edges) / num_nodes
        return MetricResult(name="avg_degree", value=avg_degree)

    @staticmethod
    def compute_connectivity(edges: list[tuple], num_nodes: int) -> MetricResult:
        if num_nodes == 0 or not edges:
            return MetricResult(name="connectivity", value=0.0)
        nodes_in_edges = set()
        for src, dst in edges:
            nodes_in_edges.add(src)
            nodes_in_edges.add(dst)
        connectivity = len(nodes_in_edges) / num_nodes if num_nodes > 0 else 0.0
        return MetricResult(name="connectivity", value=connectivity)

    @staticmethod
    def compute_avg_weight(edges: list[dict]) -> MetricResult:
        if not edges:
            return MetricResult(name="avg_weight", value=0.0)
        weights = [e.get("weight", 1.0) for e in edges]
        avg_weight = sum(weights) / len(weights)
        return MetricResult(name="avg_weight", value=avg_weight)


class CircuitMetrics:
    @staticmethod
    def compute_circuit_strength(features: list[tuple[int, int]]) -> MetricResult:
        if not features:
            return MetricResult(name="circuit_strength", value=0.0)
        return MetricResult(name="circuit_strength", value=len(features))

    @staticmethod
    def compute_feature_overlap(
        circuit1: list[tuple[int, int]], circuit2: list[tuple[int, int]]
    ) -> MetricResult:
        if not circuit1 or not circuit2:
            return MetricResult(name="feature_overlap", value=0.0)
        set1 = set(circuit1)
        set2 = set(circuit2)
        overlap = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = overlap / union if union > 0 else 0.0
        return MetricResult(name="feature_overlap", value=jaccard)

    @staticmethod
    def compute_layer_spread(features: list[tuple[int, int]]) -> MetricResult:
        if not features:
            return MetricResult(name="layer_spread", value=0.0)
        layers = set(f[0] for f in features)
        return MetricResult(name="layer_spread", value=len(layers))


class AttentionMetrics:
    @staticmethod
    def compute_attention_entropy(attention_weights: list[list[float]]) -> MetricResult:
        if not attention_weights:
            return MetricResult(name="attention_entropy", value=0.0)
        total_entropy = 0.0
        count = 0
        for row in attention_weights:
            row_entropy = 0.0
            for weight in row:
                if weight > 1e-6:
                    row_entropy -= weight * math.log2(weight)
            total_entropy += row_entropy
            count += 1
        avg_entropy = total_entropy / count if count > 0 else 0.0
        return MetricResult(name="attention_entropy", value=avg_entropy, unit="bits")

    @staticmethod
    def compute_attention_diversity(attention_weights: list[list[float]]) -> MetricResult:
        if not attention_weights:
            return MetricResult(name="attention_diversity", value=0.0)
        num_high_attn = 0
        total = 0
        for row in attention_weights:
            for weight in row:
                total += 1
                if weight > 0.1:
                    num_high_attn += 1
        diversity = num_high_attn / total if total > 0 else 0.0
        return MetricResult(name="attention_diversity", value=diversity * 100, unit="%")

    @staticmethod
    def compute_attention_concentration(attention_weights: list[list[float]]) -> MetricResult:
        if not attention_weights:
            return MetricResult(name="attention_concentration", value=0.0)
        max_weights = []
        for row in attention_weights:
            if row:
                max_weights.append(max(row))
        avg_max = sum(max_weights) / len(max_weights) if max_weights else 0.0
        return MetricResult(name="attention_concentration", value=avg_max * 100, unit="%")


class ComparisonMetrics:
    @staticmethod
    def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> MetricResult:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return MetricResult(name="cosine_similarity", value=0.0)
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return MetricResult(name="cosine_similarity", value=0.0)
        similarity = dot / (norm1 * norm2)
        return MetricResult(name="cosine_similarity", value=similarity)

    @staticmethod
    def compute_euclidean_distance(vec1: list[float], vec2: list[float]) -> MetricResult:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return MetricResult(name="euclidean_distance", value=0.0)
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=True)))
        return MetricResult(name="euclidean_distance", value=distance)

    @staticmethod
    def compute_kl_divergence(p: list[float], q: list[float]) -> MetricResult:
        if not p or not q or len(p) != len(q):
            return MetricResult(name="kl_divergence", value=0.0)
        kl = 0.0
        for pi, qi in zip(p, q, strict=True):
            if pi > 1e-6 and qi > 1e-6:
                kl += pi * math.log(pi / qi)
        return MetricResult(name="kl_divergence", value=kl)


class Aggregator:
    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)

    def add(self, metric: MetricResult) -> None:
        self.metrics[metric.name].append(metric.value)

    def add_dict(self, metrics: dict[str, MetricResult]) -> None:
        for metric in metrics.values():
            self.add(metric)

    def summarize(self) -> dict[str, MetricResult]:
        results = {}
        for name, values in self.metrics.items():
            if values:
                results[f"{name}_mean"] = MetricResult(
                    name=f"{name}_mean", value=sum(values) / len(values)
                )
                results[f"{name}_sum"] = MetricResult(
                    name=f"{name}_sum", value=sum(values)
                )
                results[f"{name}_min"] = MetricResult(
                    name=f"{name}_min", value=min(values)
                )
                results[f"{name}_max"] = MetricResult(
                    name=f"{name}_max", value=max(values)
                )
        return results
