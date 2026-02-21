from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.logit_analyzer import (
    PredictionResult,
    LogitStats,
    LogitAnalyzer,
    PredictionStabilityAnalyzer,
)


class TestPredictionResult:
    def test_result_creation(self):
        result = PredictionResult(
            token_id=123,
            token_text="hello",
            logit=5.0,
            probability=0.8,
            rank=1,
        )
        assert result.token_id == 123
        assert result.token_text == "hello"
        assert result.rank == 1


class TestLogitStats:
    def test_stats_creation(self):
        stats = LogitStats(
            mean_logit=0.0,
            std_logit=1.0,
            max_logit=5.0,
            min_logit=-5.0,
            entropy=4.5,
        )
        assert stats.mean_logit == 0.0
        assert stats.entropy == 4.5


class TestLogitAnalyzer:
    def test_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = LogitAnalyzer(mock_manager)
        assert analyzer.model_manager is mock_manager

    def test_get_top_predictions(self):
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
        mock_model.return_value = mock_output

        analyzer = LogitAnalyzer(mock_manager)

        with patch.object(analyzer, "get_logits") as mock_logits:
            mock_logits.return_value = torch.randn(50257)

            mock_tokenizer.decode.return_value = "test"

            preds = analyzer.get_top_predictions("test prompt", top_k=5)
            assert len(preds) == 5
            assert all(isinstance(p, PredictionResult) for p in preds)

    def test_compute_logit_stats(self):
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
        mock_model.return_value = mock_output

        analyzer = LogitAnalyzer(mock_manager)

        with patch.object(analyzer, "get_logits") as mock_logits:
            mock_logits.return_value = torch.randn(50257)

            stats = analyzer.compute_logit_stats("test prompt")
            assert isinstance(stats, LogitStats)

    def test_compute_prediction_confidence(self):
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
        mock_model.return_value = mock_output

        analyzer = LogitAnalyzer(mock_manager)

        with patch.object(analyzer, "get_logits") as mock_logits:
            mock_logits.return_value = torch.randn(50257)

            confidence = analyzer.compute_prediction_confidence("test prompt")
            assert 0.0 <= confidence <= 1.0


class TestPredictionStabilityAnalyzer:
    def test_stability_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = PredictionStabilityAnalyzer(mock_manager)
        assert isinstance(analyzer.analyzer, LogitAnalyzer)

    def test_measure_stability(self):
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
        mock_model.return_value = mock_output

        analyzer = PredictionStabilityAnalyzer(mock_manager)

        with patch.object(
            analyzer.analyzer, "get_top_predictions"
        ) as mock_preds:
            mock_preds.return_value = [
                PredictionResult(
                    token_id=123, token_text="test", logit=1.0, probability=0.9, rank=1
                )
            ]

            stabilities = analyzer.measure_stability(["prompt1", "prompt2"])
            assert len(stabilities) == 2

    def test_find_consistent_predictions(self):
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
        mock_model.return_value = mock_output

        analyzer = PredictionStabilityAnalyzer(mock_manager)

        with patch.object(
            analyzer.analyzer, "get_top_predictions"
        ) as mock_preds:
            mock_preds.return_value = [
                PredictionResult(
                    token_id=123, token_text="test", logit=1.0, probability=0.9, rank=1
                )
            ]

            consistent = analyzer.find_consistent_predictions(
                ["prompt1", "prompt2"], stability_threshold=0.9
            )
            assert isinstance(consistent, list)
