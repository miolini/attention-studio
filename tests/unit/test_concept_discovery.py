import pytest
import numpy as np
from attention_studio.core.concept_discovery import (
    Concept,
    ConceptDiscovery,
    ConceptAnalyzer,
    ConceptMatcher,
    ConceptVisualizer,
)


@pytest.fixture
def sample_concepts():
    return [
        Concept(
            id=0,
            name="concept_0",
            feature_indices=[0, 1, 2],
            activation_pattern=np.array([1.0, 2.0, 3.0]),
            examples=["example1", "example2"],
            confidence=0.8,
        ),
        Concept(
            id=1,
            name="concept_1",
            feature_indices=[2, 3, 4],
            activation_pattern=np.array([2.0, 3.0, 4.0]),
            examples=["example3"],
            confidence=0.7,
        ),
        Concept(
            id=2,
            name="concept_2",
            feature_indices=[5, 6, 7],
            activation_pattern=np.array([5.0, 6.0, 7.0]),
            examples=[],
            confidence=0.9,
        ),
    ]


class TestConceptDiscovery:
    def test_concept_creation(self):
        concept = Concept(
            id=0,
            name="test",
            feature_indices=[1, 2, 3],
            activation_pattern=np.array([1.0, 2.0]),
            examples=["test"],
            confidence=0.5,
        )
        assert concept.id == 0
        assert concept.name == "test"


class TestConceptAnalyzer:
    def test_compute_concept_similarity(self, sample_concepts):
        analyzer = ConceptAnalyzer(sample_concepts)
        sim = analyzer.compute_concept_similarity(sample_concepts[0], sample_concepts[1])
        assert 0 <= sim <= 1

    def test_find_related_concepts(self, sample_concepts):
        analyzer = ConceptAnalyzer(sample_concepts)
        related = analyzer.find_related_concepts(sample_concepts[0], threshold=0.1)
        assert isinstance(related, list)

    def test_compute_diversity(self, sample_concepts):
        analyzer = ConceptAnalyzer(sample_concepts)
        diversity = analyzer.compute_concept_diversity()
        assert 0 <= diversity <= 1


class TestConceptMatcher:
    def test_find_concepts_for_features(self, sample_concepts):
        matcher = ConceptMatcher(sample_concepts)
        concepts = matcher.find_concepts_for_features([0, 1, 2])
        assert isinstance(concepts, list)

    def test_match_concepts_to_tokens(self, sample_concepts):
        matcher = ConceptMatcher(sample_concepts)
        results = matcher.match_concepts_to_tokens(
            ["hello", "world"],
            [[0, 1], [2, 3]],
        )
        assert isinstance(results, dict)


class TestConceptVisualizer:
    def test_get_concept_heatmap_data(self, sample_concepts):
        data = ConceptVisualizer.get_concept_heatmap_data(sample_concepts)
        assert isinstance(data, np.ndarray)

    def test_get_concept_network_data(self, sample_concepts):
        data = ConceptVisualizer.get_concept_network_data(sample_concepts, similarity_threshold=0.1)
        assert "nodes" in data
        assert "edges" in data
