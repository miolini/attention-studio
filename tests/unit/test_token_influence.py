from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.token_influence import (
    TokenInfluence,
    InfluenceReport,
    TokenInfluenceTracker,
    InfluenceVisualizer,
)


class TestTokenInfluence:
    def test_influence_creation(self):
        influence = TokenInfluence(
            token_id=123,
            token_text="hello",
            position=5,
            influence_score=0.8,
        )
        assert influence.token_id == 123
        assert influence.position == 5


class TestInfluenceReport:
    def test_report_creation(self):
        influence = TokenInfluence(
            token_id=123,
            token_text="hello",
            position=5,
            influence_score=0.8,
        )
        report = InfluenceReport(
            prompt="test prompt",
            target_position=10,
            tokens=[influence],
            total_influence=0.8,
        )
        assert report.prompt == "test prompt"
        assert len(report.tokens) == 1


class TestTokenInfluenceTracker:
    def test_tracker_initialization(self):
        mock_manager = MagicMock()
        tracker = TokenInfluenceTracker(mock_manager)
        assert tracker.model_manager is mock_manager

    def test_compute_influence_by_gradient(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_logits = torch.randn(1, 10, 50257)
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output

        mock_model.transformer.h = [MagicMock() for _ in range(12)]
        for layer in mock_model.transformer.h:
            layer.weight = MagicMock()
            layer.weight.grad = torch.randn(768, 768)

        tracker = TokenInfluenceTracker(mock_manager)

        with patch.object(torch.Tensor, "backward", return_value=None):
            result = tracker.compute_influence_by_gradient("test prompt")
            assert isinstance(result, dict)


class TestInfluenceVisualizer:
    def test_visualizer_initialization(self):
        mock_manager = MagicMock()
        tracker = TokenInfluenceTracker(mock_manager)
        visualizer = InfluenceVisualizer(tracker)
        assert visualizer.tracker is tracker

    def test_get_top_influential_tokens(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 50257)
        mock_output.hidden_states = [torch.randn(1, 10, 768) for _ in range(12)]
        mock_model.return_value = mock_output

        tracker = TokenInfluenceTracker(mock_manager)
        visualizer = InfluenceVisualizer(tracker)

        with patch.object(
            tracker, "compute_influence_by_ablation"
        ) as mock_influence:
            mock_influence.return_value = InfluenceReport(
                prompt="test",
                target_position=5,
                tokens=[
                    TokenInfluence(1, "tok", 0, 0.5),
                    TokenInfluence(2, "tok2", 1, 0.3),
                ],
                total_influence=0.8,
            )

            result = visualizer.get_top_influential_tokens("test prompt", top_k=2)
            assert len(result) <= 2
