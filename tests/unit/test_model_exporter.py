from __future__ import annotations

import torch
import pytest
from pathlib import Path
import tempfile

from attention_studio.core.model_exporter import (
    ExportResult,
    ModelExporter,
    BatchExporter,
)


class TestExportResult:
    def test_result_creation(self):
        result = ExportResult(
            success=True,
            output_path="/output/model.pt",
            format="torchscript",
        )
        assert result.success is True
        assert result.format == "torchscript"


class TestModelExporter:
    def test_exporter_creation(self):
        model = torch.nn.Linear(10, 10)
        exporter = ModelExporter(model)
        assert exporter.model is model

    def test_export_to_state_dict(self):
        model = torch.nn.Linear(10, 10)
        exporter = ModelExporter(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model")
            result = exporter.export_to_state_dict(output_path)

            assert result.success is True
            assert result.format == "state_dict"
            assert Path(result.output_path).exists()

    def test_export_to_safetensors(self):
        model = torch.nn.Linear(10, 10)
        exporter = ModelExporter(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "model")
            result = exporter.export_to_safetensors(output_path)

            if result.success:
                assert result.format == "safetensors"
                assert Path(result.output_path).exists()

    def test_export_unsupported_format(self):
        model = torch.nn.Linear(10, 10)
        exporter = ModelExporter(model)

        result = exporter.export_model("output", "unsupported_format")

        assert result.success is False
        assert "Unsupported format" in result.error


class TestBatchExporter:
    def test_batch_exporter_creation(self):
        models = [torch.nn.Linear(10, 10) for _ in range(3)]
        exporter = BatchExporter(models)
        assert len(exporter.models) == 3

    def test_export_all(self):
        models = [torch.nn.Linear(10, 10) for _ in range(2)]
        exporter = BatchExporter(models)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = exporter.export_all(tmpdir, "state_dict", name_prefix="model")

            assert len(results) == 2
            assert all(r.success for r in results)
