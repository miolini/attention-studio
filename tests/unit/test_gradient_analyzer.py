from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.gradient_analyzer import (
    GradientStats,
    GradientAnalyzer,
    GradientFlowTracker,
)


class TestGradientStats:
    def test_stats_creation(self):
        stats = GradientStats(
            layer_idx=5,
            mean_grad_norm=0.1,
            max_grad_norm=0.5,
            grad_std=0.05,
            zero_fraction=0.2,
        )
        assert stats.layer_idx == 5
        assert stats.mean_grad_norm == 0.1


class TestGradientAnalyzer:
    def test_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = GradientAnalyzer(mock_manager)
        assert analyzer.model_manager is mock_manager

    def test_compute_layer_gradient_stats(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        analyzer = GradientAnalyzer(mock_manager)

        with patch.object(analyzer, "compute_gradients") as mock_grads:
            mock_grads.return_value = {
                0: torch.randn(768, 768),
                1: torch.randn(768, 768),
            }

            stats = analyzer.compute_layer_gradient_stats("test prompt")
            assert len(stats) > 0

    def test_find_gradient_hero_layers(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        analyzer = GradientAnalyzer(mock_manager)

        with patch.object(analyzer, "compute_gradients") as mock_grads:
            mock_grads.return_value = {
                0: torch.randn(768, 768),
                1: torch.randn(768, 768),
            }

            heroes = analyzer.find_gradient_hero_layers(["prompt1", "prompt2"])
            assert len(heroes) <= 5


class TestGradientFlowTracker:
    def test_tracker_initialization(self):
        mock_manager = MagicMock()
        tracker = GradientFlowTracker(mock_manager)
        assert isinstance(tracker.analyzer, GradientAnalyzer)
        assert tracker.history == []

    def test_record(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        tracker = GradientFlowTracker(mock_manager)

        with patch.object(tracker.analyzer, "compute_gradients") as mock_grads:
            mock_grads.return_value = {
                0: torch.randn(768, 768),
                1: torch.randn(768, 768),
            }

            tracker.record("test prompt")
            assert len(tracker.history) == 1

    def test_get_layer_history(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        tracker = GradientFlowTracker(mock_manager)

        tracker.history = [
            {0: 1.0, 1: 2.0},
            {0: 1.5, 1: 2.5},
            {0: 1.2, 1: 2.2},
        ]

        history = tracker.get_layer_history(0)
        assert len(history) == 3
        assert history == [1.0, 1.5, 1.2]

    def test_compute_stability(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        tracker = GradientFlowTracker(mock_manager)

        tracker.history = [
            {0: 1.0, 1: 2.0},
            {0: 1.5, 1: 2.5},
            {0: 1.2, 1: 2.2},
        ]

        stability = tracker.compute_stability(0)
        assert stability >= 0.0

    def test_clear(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        tracker = GradientFlowTracker(mock_manager)

        tracker.history = [{0: 1.0}]
        tracker.clear()
        assert tracker.history == []
