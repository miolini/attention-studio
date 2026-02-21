from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.model_diagnostics import (
    DiagnosticResult,
    DiagnosticReport,
    ModelDiagnostics,
)


class TestDiagnosticResult:
    def test_result_creation(self):
        result = DiagnosticResult(
            name="Test Check",
            passed=True,
            value=0.5,
            threshold=0.1,
            message="Test passed",
        )
        assert result.name == "Test Check"
        assert result.passed is True


class TestDiagnosticReport:
    def test_report_creation(self):
        result = DiagnosticResult(
            name="Test",
            passed=True,
            value=0.5,
            threshold=0.1,
            message="OK",
        )
        report = DiagnosticReport(
            model_name="gpt2",
            results=[result],
            overall_passed=True,
        )
        assert report.model_name == "gpt2"
        assert len(report.results) == 1


class TestModelDiagnostics:
    def test_diagnostics_initialization(self):
        mock_manager = MagicMock()
        diagnostics = ModelDiagnostics(mock_manager)
        assert diagnostics.model_manager is mock_manager

    def test_check_nan_values(self):
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
        mock_output.hidden_states = None
        mock_model.return_value = mock_output

        diagnostics = ModelDiagnostics(mock_manager)

        result = diagnostics.check_nan_values(["test prompt"])
        assert isinstance(result, DiagnosticResult)
        assert result.name == "NaN Check"

    def test_check_inf_values(self):
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

        diagnostics = ModelDiagnostics(mock_manager)

        result = diagnostics.check_inf_values(["test prompt"])
        assert isinstance(result, DiagnosticResult)
        assert result.name == "Inf Check"

    def test_check_output_variance(self):
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

        diagnostics = ModelDiagnostics(mock_manager)

        result = diagnostics.check_output_variance(["test prompt"])
        assert isinstance(result, DiagnosticResult)
        assert result.name == "Output Variance"

    def test_check_token_probabilities(self):
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

        diagnostics = ModelDiagnostics(mock_manager)

        result = diagnostics.check_token_probabilities("test prompt")
        assert isinstance(result, DiagnosticResult)
        assert result.name == "Token Probability Distribution"

    def test_check_layer_consistency(self):
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
        mock_output.hidden_states = [torch.randn(1, 10, 768) for _ in range(12)]
        mock_model.return_value = mock_output

        diagnostics = ModelDiagnostics(mock_manager)

        result = diagnostics.check_layer_consistency(["test prompt"])
        assert isinstance(result, DiagnosticResult)
        assert result.name == "Layer Consistency"

    def test_run_full_diagnostics(self):
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

        mock_model.name_or_path = "test-model"

        mock_transformer = MagicMock()
        mock_wte = MagicMock()
        mock_wte.weight.data = torch.randn(50257, 768)
        mock_transformer.wte = mock_wte
        mock_model.transformer = mock_transformer

        diagnostics = ModelDiagnostics(mock_manager)

        report = diagnostics.run_full_diagnostics(["prompt1", "prompt2"])
        assert isinstance(report, DiagnosticReport)
        assert len(report.results) >= 6
