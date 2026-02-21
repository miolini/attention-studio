from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExportResult:
    success: bool
    output_path: str
    format: str
    file_size: int | None = None
    error: str | None = None


class ModelExporter:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def export_to_torchscript(
        self,
        output_path: str,
        example_input: torch.Tensor | None = None,
    ) -> ExportResult:
        try:
            self.model.eval()

            if example_input is None:
                scripted = torch.jit.script(self.model)
            else:
                scripted = torch.jit.trace(self.model, example_input)

            output_path = str(Path(output_path).with_suffix(".pt"))
            scripted.save(output_path)

            file_size = Path(output_path).stat().st_size

            return ExportResult(
                success=True,
                output_path=output_path,
                format="torchscript",
                file_size=file_size,
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="torchscript",
                error=str(e),
            )

    def export_to_state_dict(
        self,
        output_path: str,
    ) -> ExportResult:
        try:
            output_path = str(Path(output_path).with_suffix(".pt"))

            torch.save(self.model.state_dict(), output_path)

            file_size = Path(output_path).stat().st_size

            return ExportResult(
                success=True,
                output_path=output_path,
                format="state_dict",
                file_size=file_size,
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="state_dict",
                error=str(e),
            )

    def export_to_onnx(
        self,
        output_path: str,
        example_input: torch.Tensor,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict | None = None,
    ) -> ExportResult:
        try:
            self.model.eval()

            output_path = str(Path(output_path).with_suffix(".onnx"))

            if input_names is None:
                input_names = ["input"]
            if output_names is None:
                output_names = ["output"]

            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
                opset_version=14,
            )

            file_size = Path(output_path).stat().st_size

            return ExportResult(
                success=True,
                output_path=output_path,
                format="onnx",
                file_size=file_size,
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="onnx",
                error=str(e),
            )

    def export_to_huggingface(
        self,
        output_dir: str,
        config: dict[str, Any],
        tokenizer: Any | None = None,
    ) -> ExportResult:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            model_path = output_path / "pytorch_model.bin"
            torch.save(self.model.state_dict(), model_path)

            config_path = output_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            if tokenizer is not None:
                tokenizer.save_pretrained(str(output_path))

            file_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())

            return ExportResult(
                success=True,
                output_path=str(output_path),
                format="huggingface",
                file_size=file_size,
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=output_dir,
                format="huggingface",
                error=str(e),
            )

    def export_to_safetensors(
        self,
        output_path: str,
    ) -> ExportResult:
        try:
            from safetensors.torch import save_file

            output_path = str(Path(output_path).with_suffix(".safetensors"))

            state_dict = self.model.state_dict()
            save_file(state_dict, output_path)

            file_size = Path(output_path).stat().st_size

            return ExportResult(
                success=True,
                output_path=output_path,
                format="safetensors",
                file_size=file_size,
            )
        except ImportError:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="safetensors",
                error="safetensors not installed. Install with: pip install safetensors",
            )
        except Exception as e:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="safetensors",
                error=str(e),
            )

    def export_model(
        self,
        output_path: str,
        export_format: str,
        **kwargs,
    ) -> ExportResult:
        format_lower = export_format.lower()

        if format_lower == "torchscript":
            return self.export_to_torchscript(output_path, kwargs.get("example_input"))
        elif format_lower == "state_dict":
            return self.export_to_state_dict(output_path)
        elif format_lower == "onnx":
            return self.export_to_onnx(
                output_path,
                kwargs.get("example_input"),
                kwargs.get("input_names"),
                kwargs.get("output_names"),
                kwargs.get("dynamic_axes"),
            )
        elif format_lower == "huggingface":
            return self.export_to_huggingface(
                output_path,
                kwargs.get("config", {}),
                kwargs.get("tokenizer"),
            )
        elif format_lower == "safetensors":
            return self.export_to_safetensors(output_path)
        else:
            return ExportResult(
                success=False,
                output_path=output_path,
                format=format,
                error=f"Unsupported format: {format}",
            )


class BatchExporter:
    def __init__(self, models: list[torch.nn.Module]):
        self.models = models

    def export_all(
        self,
        output_dir: str,
        export_format: str,
        name_prefix: str = "model",
        **kwargs,
    ) -> list[ExportResult]:
        results = []

        for i, model in enumerate(self.models):
            exporter = ModelExporter(model)
            output_path = f"{output_dir}/{name_prefix}_{i}"
            result = exporter.export_model(output_path, export_format, **kwargs)
            results.append(result)

        return results
