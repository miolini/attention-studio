from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class Concept:
    id: int
    name: str
    feature_indices: list[int]
    activation_pattern: np.ndarray
    examples: list[str]
    confidence: float


class ConceptDiscovery:
    def __init__(self, transcoder: Any):
        self.transcoder = transcoder
        self.concepts: list[Concept] = []

    def discover_by_clustering(
        self,
        hidden_states: torch.Tensor,
        n_concepts: int = 10,
    ) -> list[Concept]:
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return []

        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        features_flat = features.view(-1, features.shape[-1]).cpu().numpy()

        if features_flat.shape[0] < n_concepts:
            n_concepts = features_flat.shape[0]

        kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_flat)

        concepts = []
        for i in range(n_concepts):
            mask = labels == i
            pattern = features_flat[mask].mean(axis=0) if mask.any() else np.zeros(features_flat.shape[1])

            feature_importance = np.abs(pattern)
            top_features = np.argsort(feature_importance)[-20:][::-1]

            concept = Concept(
                id=i,
                name=f"concept_{i}",
                feature_indices=top_features.tolist(),
                activation_pattern=pattern,
                examples=[],
                confidence=float(kmeans.cluster_centers_[i].var()),
            )
            concepts.append(concept)

        self.concepts = concepts
        return concepts

    def discover_by_activation(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 100,
    ) -> list[Concept]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        mean_activations = features.abs().mean(dim=(0, 1)).cpu().numpy()
        top_indices = np.argsort(mean_activations)[-top_k:][::-1]

        concepts = []
        for i, idx in enumerate(top_indices):
            concept = Concept(
                id=i,
                name=f"feature_{idx}",
                feature_indices=[int(idx)],
                activation_pattern=mean_activations,
                examples=[],
                confidence=float(mean_activations[idx]),
            )
            concepts.append(concept)

        self.concepts = concepts
        return concepts

    def discover_by_correlation(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 0.7,
    ) -> list[Concept]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        features_flat = features.view(-1, features.shape[-1]).cpu().numpy()
        corr_matrix = np.corrcoef(features_flat.T)

        np.fill_diagonal(corr_matrix, 0)

        concepts = []
        concept_id = 0

        visited = set()
        for i in range(corr_matrix.shape[0]):
            if i in visited:
                continue

            correlated = np.where(np.abs(corr_matrix[i]) > threshold)[0]
            if len(correlated) > 1:
                concept_features = [i] + [j for j in correlated if j not in visited]
                visited.update(concept_features)

                pattern = features_flat[:, concept_features].mean(axis=1)
                concept = Concept(
                    id=concept_id,
                    name=f"correlated_concept_{concept_id}",
                    feature_indices=concept_features,
                    activation_pattern=pattern,
                    examples=[],
                    confidence=float(np.abs(corr_matrix[i, correlated]).mean()),
                )
                concepts.append(concept)
                concept_id += 1

        self.concepts = concepts
        return concepts


class ConceptAnalyzer:
    def __init__(self, concepts: list[Concept]):
        self.concepts = concepts

    def compute_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        if len(concept1.feature_indices) == 0 or len(concept2.feature_indices) == 0:
            return 0.0

        set1 = set(concept1.feature_indices)
        set2 = set(concept2.feature_indices)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def find_related_concepts(
        self,
        concept: Concept,
        threshold: float = 0.3,
    ) -> list[tuple[Concept, float]]:
        related = []
        for other in self.concepts:
            if other.id == concept.id:
                continue
            similarity = self.compute_concept_similarity(concept, other)
            if similarity > threshold:
                related.append((other, similarity))

        return sorted(related, key=lambda x: x[1], reverse=True)

    def compute_concept_diversity(self) -> float:
        if not self.concepts:
            return 0.0

        total_unique = set()
        for concept in self.concepts:
            total_unique.update(concept.feature_indices)

        return len(total_unique) / sum(len(c.feature_indices) for c in self.concepts)


class ConceptMatcher:
    def __init__(self, concepts: list[Concept]):
        self.concepts = concepts
        self._build_index()

    def _build_index(self) -> None:
        self.feature_to_concepts: dict[int, list[int]] = {}
        for concept in self.concepts:
            for feature_idx in concept.feature_indices:
                if feature_idx not in self.feature_to_concepts:
                    self.feature_to_concepts[feature_idx] = []
                self.feature_to_concepts[feature_idx].append(concept.id)

    def find_concepts_for_features(self, feature_indices: list[int]) -> list[Concept]:
        concept_scores: dict[int, float] = {}

        for feature_idx in feature_indices:
            if feature_idx in self.feature_to_concepts:
                for concept_id in self.feature_to_concepts[feature_idx]:
                    concept_scores[concept_id] = concept_scores.get(concept_id, 0) + 1

        results = []
        for concept_id, score in concept_scores.items():
            concept = next((c for c in self.concepts if c.id == concept_id), None)
            if concept:
                results.append((concept, score / len(feature_indices)))

        return [c for c, _ in sorted(results, key=lambda x: x[1], reverse=True)]

    def match_concepts_to_tokens(
        self,
        tokens: list[str],
        token_features: list[list[int]],
    ) -> dict[str, list[Concept]]:
        results = {}
        for token, features in zip(tokens, token_features, strict=True):
            matched_concepts = self.find_concepts_for_features(features)
            results[token] = matched_concepts[:5]

        return results


class ConceptVisualizer:
    @staticmethod
    def get_concept_heatmap_data(concepts: list[Concept]) -> np.ndarray:
        if not concepts:
            return np.array([])

        max_features = max(len(c.feature_indices) for c in concepts)
        data = np.zeros((len(concepts), max_features))

        for i, concept in enumerate(concepts):
            for j, feat_idx in enumerate(concept.feature_indices):
                data[i, j] = concept.activation_pattern[feat_idx] if feat_idx < len(concept.activation_pattern) else 0

        return data

    @staticmethod
    def get_concept_network_data(
        concepts: list[Concept],
        similarity_threshold: float = 0.3,
    ) -> dict[str, Any]:
        nodes = [{"id": c.id, "label": c.name, "size": len(c.feature_indices)} for c in concepts]

        edges = []
        analyzer = ConceptAnalyzer(concepts)

        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                sim = analyzer.compute_concept_similarity(c1, c2)
                if sim > similarity_threshold:
                    edges.append({"source": c1.id, "target": c2.id, "weight": sim})

        return {"nodes": nodes, "edges": edges}
