from __future__ import annotations

import numpy as np
import torch
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.representation_analyzer import (
    RepresentationMetrics,
    RepresentationAnalyzer,
    RepresentationComparator,
)


class TestRepresentationMetrics:
    def test_metrics_creation(self):
        metrics = RepresentationMetrics(
            layer_idx=5,
            mean_norm=1.5,
            std_norm=0.3,
            variance_explained=0.8,
            rank=768,
        )
        assert metrics.layer_idx == 5
        assert metrics.mean_norm == 1.5


class TestRepresentationAnalyzer:
    def test_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = RepresentationAnalyzer(mock_manager)
        assert analyzer.model_manager is mock_manager

    def test_compute_representation_norms(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_tokenizer = mock_manager.tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.hidden_states = [
            torch.randn(1, 10, 768) for _ in range(12)
        ]
        mock_model.return_value = mock_output

        analyzer = RepresentationAnalyzer(mock_manager)

        with patch.object(analyzer, "get_hidden_states") as mock_states:
            mock_states.return_value = {0: torch.randn(5, 768)}

            norms = analyzer.compute_representation_norms("test prompt")
            assert isinstance(norms, dict)

    def test_compute_pca(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_tokenizer = mock_manager.tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.hidden_states = [
            torch.randn(1, 10, 768) for _ in range(12)
        ]
        mock_model.return_value = mock_output

        analyzer = RepresentationAnalyzer(mock_manager)

        with patch.object(analyzer, "get_hidden_states") as mock_states:
            mock_states.return_value = {
                0: torch.randn(5, 768)
            }

            result = analyzer.compute_pca(["prompt1", "prompt2"], 0, n_components=5)
            assert "components" in result
            assert "explained_variance" in result
            assert result["components"].shape[0] == 5

    def test_compute_metrics(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_tokenizer = mock_manager.tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.hidden_states = [
            torch.randn(1, 10, 768) for _ in range(12)
        ]
        mock_model.return_value = mock_output

        analyzer = RepresentationAnalyzer(mock_manager)

        with patch.object(analyzer, "get_hidden_states") as mock_states:
            mock_states.return_value = {
                0: torch.randn(5, 768)
            }

            metrics = analyzer.compute_metrics(["prompt1", "prompt2"], 0)
            assert isinstance(metrics, RepresentationMetrics)
            assert metrics.layer_idx == 0


class TestRepresentationComparator:
    def test_comparator_initialization(self):
        mock_manager = MagicMock()
        comparator = RepresentationComparator(mock_manager)
        assert isinstance(comparator.analyzer, RepresentationAnalyzer)

    def test_compare_prompts(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_tokenizer = mock_manager.tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.hidden_states = [
            torch.randn(1, 10, 768) for _ in range(12)
        ]
        mock_model.return_value = mock_output

        comparator = RepresentationComparator(mock_manager)

        with patch.object(
            comparator.analyzer, "compute_representation_similarity"
        ) as mock_sim:
            mock_sim.return_value = 0.85

            similarities = comparator.compare_prompts("prompt_a", "prompt_b")
            assert isinstance(similarities, dict)

    def test_find_similar_prompts(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_tokenizer = mock_manager.tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.hidden_states = [
            torch.randn(1, 10, 768) for _ in range(12)
        ]
        mock_model.return_value = mock_output

        comparator = RepresentationComparator(mock_manager)

        with patch.object(
            comparator.analyzer, "compute_representation_similarity"
        ) as mock_sim:
            mock_sim.return_value = 0.75

            results = comparator.find_similar_prompts(
                "query", ["candidate1", "candidate2", "candidate3"], 0
            )
            assert len(results) <= 3
