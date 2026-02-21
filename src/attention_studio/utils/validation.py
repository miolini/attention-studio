from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def valid(cls, warnings: Optional[list[str]] = None) -> ValidationResult:
        return cls(is_valid=True, errors=[], warnings=warnings or [])

    @classmethod
    def invalid(cls, errors: list[str], warnings: Optional[list[str]] = None) -> ValidationResult:
        return cls(is_valid=False, errors=errors, warnings=warnings or [])

    def merge(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


class PromptValidator:
    MAX_LENGTH = 10000
    MIN_LENGTH = 1

    def __init__(self, max_length: Optional[int] = None, min_length: Optional[int] = None):
        self.max_length = max_length or self.MAX_LENGTH
        self.min_length = min_length or self.MIN_LENGTH

    def validate(self, prompt: str) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(prompt, str):
            errors.append("Prompt must be a string")
            return ValidationResult.invalid(errors)

        if len(prompt) < self.min_length:
            errors.append(f"Prompt too short (min: {self.min_length})")
        elif len(prompt) > self.max_length:
            errors.append(f"Prompt too long (max: {self.max_length})")

        if prompt.strip() != prompt:
            warnings.append("Prompt has leading/trailing whitespace")

        null_chars = [i for i, c in enumerate(prompt) if ord(c) == 0]
        if null_chars:
            errors.append(f"Prompt contains null characters at positions: {null_chars}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch(self, prompts: list[str]) -> ValidationResult:
        results = [self.validate(p) for p in prompts]
        all_errors = []
        all_warnings = []
        for i, result in enumerate(results):
            for error in result.errors:
                all_errors.append(f"Prompt {i}: {error}")
            for warning in result.warnings:
                all_warnings.append(f"Prompt {i}: {warning}")
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )


class ModelConfigValidator:
    VALID_DEVICES = {"cuda", "cpu", "mps"}
    VALID_DTYPES = {"float32", "float16", "bfloat16", "float64"}

    def __init__(
        self,
        valid_devices: Optional[list[str]] = None,
        valid_dtypes: Optional[list[str]] = None,
    ):
        self.valid_devices = set(valid_devices) if valid_devices else self.VALID_DEVICES
        self.valid_dtypes = set(valid_dtypes) if valid_dtypes else self.VALID_DTYPES

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return ValidationResult.invalid(errors)

        if "name" in config:
            name = config["name"]
            if not isinstance(name, str) or not name:
                errors.append("'name' must be a non-empty string")
        else:
            errors.append("'name' is required")

        if "device" in config:
            device = config["device"]
            if device not in self.valid_devices:
                errors.append(f"Invalid device '{device}'. Valid: {self.valid_devices}")
        else:
            warnings.append("No device specified, default will be used")

        if "dtype" in config:
            dtype = config["dtype"]
            if dtype not in self.valid_dtypes:
                errors.append(f"Invalid dtype '{dtype}'. Valid: {self.valid_dtypes}")

        if "max_length" in config:
            max_length = config["max_length"]
            if not isinstance(max_length, int) or max_length <= 0:
                errors.append("'max_length' must be a positive integer")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class FeatureIndexValidator:
    def __init__(self, max_features: int):
        self.max_features = max_features

    def validate(self, feature_idx: int) -> ValidationResult:
        errors = []

        if not isinstance(feature_idx, int):
            errors.append("Feature index must be an integer")
        elif feature_idx < 0:
            errors.append(f"Feature index must be non-negative (got {feature_idx})")
        elif feature_idx >= self.max_features:
            errors.append(f"Feature index out of range (max: {self.max_features - 1})")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[],
        )

    def validate_batch(self, feature_indices: list[int]) -> ValidationResult:
        results = [self.validate(idx) for idx in feature_indices]
        all_errors = []
        for i, result in enumerate(results):
            for error in result.errors:
                all_errors.append(f"Index {i}: {error}")
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=[],
        )


class LayerIndexValidator:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers

    def validate(self, layer_idx: int) -> ValidationResult:
        errors = []

        if not isinstance(layer_idx, int):
            errors.append("Layer index must be an integer")
        elif layer_idx < 0:
            errors.append(f"Layer index must be non-negative (got {layer_idx})")
        elif layer_idx >= self.num_layers:
            errors.append(f"Layer index out of range (max: {self.num_layers - 1})")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[],
        )

    def validate_batch(self, layer_indices: list[int]) -> ValidationResult:
        results = [self.validate(idx) for idx in layer_indices]
        all_errors = []
        for i, result in enumerate(results):
            for error in result.errors:
                all_errors.append(f"Layer {i}: {error}")
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=[],
        )


class PathValidator:
    def __init__(
        self,
        must_exist: bool = False,
        must_be_dir: bool = False,
        must_be_file: bool = False,
        allowed_extensions: Optional[list[str]] = None,
    ):
        self.must_exist = must_exist
        self.must_be_dir = must_be_dir
        self.must_be_file = must_be_file
        self.allowed_extensions = allowed_extensions

    def validate(self, path: str) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(path, (str, Path)):
            errors.append("Path must be a string or Path")
            return ValidationResult.invalid(errors)

        path_obj = Path(path)

        if self.must_exist and not path_obj.exists():
            errors.append(f"Path does not exist: {path}")

        if self.must_be_dir and path_obj.exists() and not path_obj.is_dir():
            errors.append(f"Path is not a directory: {path}")

        if self.must_be_file and path_obj.exists() and not path_obj.is_file():
            errors.append(f"Path is not a file: {path}")

        if self.allowed_extensions and path_obj.suffix:
            if path_obj.suffix not in self.allowed_extensions:
                errors.append(
                    f"Invalid extension '{path_obj.suffix}'. "
                    f"Allowed: {self.allowed_extensions}"
                )

        try:
            if path_obj.exists() and path_obj.is_file():
                if path_obj.stat().st_size == 0:
                    warnings.append("File is empty")
        except OSError:
            pass

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class ThresholdValidator:
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, inclusive: bool = True):
        self.min_val = min_val
        self.max_val = max_val
        self.inclusive = inclusive

    def validate(self, value: float, name: str = "value") -> ValidationResult:
        errors = []

        if not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number")
        elif self.inclusive:
            if value < self.min_val or value > self.max_val:
                errors.append(f"{name} must be between {self.min_val} and {self.max_val}")
        else:
            if value <= self.min_val or value >= self.max_val:
                errors.append(f"{name} must be strictly between {self.min_val} and {self.max_val}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[],
        )


class HyperparameterValidator:
    def __init__(self):
        pass

    def validate_learning_rate(self, lr: float) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(lr, (int, float)):
            errors.append("Learning rate must be a number")
        elif lr <= 0:
            errors.append("Learning rate must be positive")
        elif lr < 1e-6:
            warnings.append("Learning rate very small, training may be slow")
        elif lr > 1:
            errors.append("Learning rate too large, training may diverge")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch_size(self, batch_size: int) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(batch_size, int):
            errors.append("Batch size must be an integer")
        elif batch_size <= 0:
            errors.append("Batch size must be positive")
        elif batch_size > 512:
            warnings.append("Batch size very large, may cause OOM")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_epochs(self, epochs: int) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(epochs, int):
            errors.append("Epochs must be an integer")
        elif epochs <= 0:
            errors.append("Epochs must be positive")
        elif epochs > 1000:
            warnings.append("Large number of epochs, training may take long")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


def sanitize_prompt(prompt: str) -> str:
    prompt = prompt.replace("\x00", "")
    prompt = prompt.strip()
    return prompt


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = filename.strip(". ")
    if not filename:
        filename = "unnamed"
    return filename


def is_safe_path(path: Path, base_dir: Path) -> bool:
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        return str(resolved).startswith(str(base_resolved))
    except (ValueError, OSError):
        return False
