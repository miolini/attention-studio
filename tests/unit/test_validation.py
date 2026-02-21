import pytest
from attention_studio.utils.validation import (
    PromptValidator,
    ModelConfigValidator,
    FeatureIndexValidator,
    LayerIndexValidator,
    PathValidator,
    ThresholdValidator,
    HyperparameterValidator,
    sanitize_prompt,
    sanitize_filename,
    is_safe_path,
    ValidationResult,
)


class TestValidationResult:
    def test_valid_result(self):
        result = ValidationResult.valid()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_valid_result_with_warnings(self):
        result = ValidationResult.valid(warnings=["test warning"])
        assert result.is_valid is True
        assert result.warnings == ["test warning"]

    def test_invalid_result(self):
        result = ValidationResult.invalid(["error1", "error2"])
        assert result.is_valid is False
        assert result.errors == ["error1", "error2"]

    def test_merge_valid_results(self):
        r1 = ValidationResult.valid()
        r2 = ValidationResult.valid(warnings=["w1"])
        merged = r1.merge(r2)
        assert merged.is_valid is True
        assert merged.warnings == ["w1"]

    def test_merge_invalid_results(self):
        r1 = ValidationResult.invalid(["e1"])
        r2 = ValidationResult.valid(warnings=["w1"])
        merged = r1.merge(r2)
        assert merged.is_valid is False
        assert merged.errors == ["e1"]
        assert merged.warnings == ["w1"]


class TestPromptValidator:
    def test_valid_prompt(self):
        validator = PromptValidator()
        result = validator.validate("Hello, world!")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_empty_prompt(self):
        validator = PromptValidator(min_length=1)
        result = validator.validate("")
        assert result.is_valid is False

    def test_prompt_too_long(self):
        validator = PromptValidator(max_length=10)
        result = validator.validate("This is a very long prompt")
        assert result.is_valid is False

    def test_prompt_with_whitespace(self):
        validator = PromptValidator()
        result = validator.validate("  Hello  ")
        assert len(result.warnings) > 0

    def test_prompt_with_null_char(self):
        validator = PromptValidator()
        result = validator.validate("Hello\x00World")
        assert result.is_valid is False

    def test_prompt_not_string(self):
        validator = PromptValidator()
        result = validator.validate(123)
        assert result.is_valid is False

    def test_batch_validation(self):
        validator = PromptValidator(max_length=20)
        result = validator.validate_batch(["short", "a" * 100])
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestModelConfigValidator:
    def test_valid_config(self):
        validator = ModelConfigValidator()
        result = validator.validate({"name": "gpt2", "device": "cuda"})
        assert result.is_valid is True

    def test_missing_name(self):
        validator = ModelConfigValidator()
        result = validator.validate({"device": "cuda"})
        assert result.is_valid is False

    def test_invalid_device(self):
        validator = ModelConfigValidator()
        result = validator.validate({"name": "gpt2", "device": "invalid"})
        assert result.is_valid is False

    def test_invalid_dtype(self):
        validator = ModelConfigValidator()
        result = validator.validate({"name": "gpt2", "dtype": "invalid"})
        assert result.is_valid is False

    def test_invalid_max_length(self):
        validator = ModelConfigValidator()
        result = validator.validate({"name": "gpt2", "max_length": -1})
        assert result.is_valid is False

    def test_config_not_dict(self):
        validator = ModelConfigValidator()
        result = validator.validate("not a dict")
        assert result.is_valid is False


class TestFeatureIndexValidator:
    def test_valid_index(self):
        validator = FeatureIndexValidator(max_features=100)
        result = validator.validate(50)
        assert result.is_valid is True

    def test_negative_index(self):
        validator = FeatureIndexValidator(max_features=100)
        result = validator.validate(-1)
        assert result.is_valid is False

    def test_index_out_of_range(self):
        validator = FeatureIndexValidator(max_features=100)
        result = validator.validate(100)
        assert result.is_valid is False

    def test_batch_validation(self):
        validator = FeatureIndexValidator(max_features=100)
        result = validator.validate_batch([10, 50, 200])
        assert result.is_valid is False


class TestLayerIndexValidator:
    def test_valid_layer(self):
        validator = LayerIndexValidator(num_layers=12)
        result = validator.validate(5)
        assert result.is_valid is True

    def test_negative_layer(self):
        validator = LayerIndexValidator(num_layers=12)
        result = validator.validate(-1)
        assert result.is_valid is False

    def test_layer_out_of_range(self):
        validator = LayerIndexValidator(num_layers=12)
        result = validator.validate(12)
        assert result.is_valid is False


class TestPathValidator:
    def test_valid_path_no_check(self):
        validator = PathValidator()
        result = validator.validate("/some/path")
        assert result.is_valid is True

    def test_path_must_exist(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            validator = PathValidator(must_exist=True)
            result = validator.validate(temp_path)
            assert result.is_valid is True
            result = validator.validate("/nonexistent/path")
            assert result.is_valid is False
        finally:
            os.unlink(temp_path)

    def test_path_must_be_file(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            validator = PathValidator(must_be_file=True)
            result = validator.validate(temp_path)
            assert result.is_valid is True
        finally:
            os.unlink(temp_path)

    def test_allowed_extensions(self):
        validator = PathValidator(allowed_extensions=[".json", ".txt"])
        result = validator.validate("data.json")
        assert result.is_valid is True
        result = validator.validate("data.csv")
        assert result.is_valid is False


class TestThresholdValidator:
    def test_valid_threshold(self):
        validator = ThresholdValidator(min_val=0.0, max_val=1.0)
        result = validator.validate(0.5)
        assert result.is_valid is True

    def test_threshold_out_of_range(self):
        validator = ThresholdValidator(min_val=0.0, max_val=1.0)
        result = validator.validate(1.5)
        assert result.is_valid is False


class TestHyperparameterValidator:
    def test_valid_learning_rate(self):
        validator = HyperparameterValidator()
        result = validator.validate_learning_rate(0.001)
        assert result.is_valid is True

    def test_invalid_learning_rate(self):
        validator = HyperparameterValidator()
        result = validator.validate_learning_rate(-0.001)
        assert result.is_valid is False

    def test_valid_batch_size(self):
        validator = HyperparameterValidator()
        result = validator.validate_batch_size(32)
        assert result.is_valid is True

    def test_invalid_batch_size(self):
        validator = HyperparameterValidator()
        result = validator.validate_batch_size(-1)
        assert result.is_valid is False


class TestSanitizeFunctions:
    def test_sanitize_prompt(self):
        assert sanitize_prompt("  hello  ") == "hello"
        assert sanitize_prompt("hello\x00world") == "helloworld"

    def test_sanitize_filename(self):
        assert sanitize_filename("my<file>.txt") == "my_file_.txt"
        assert sanitize_filename("  ") == "unnamed"

    def test_is_safe_path(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            safe = base / "subdir"
            safe.mkdir()
            assert is_safe_path(safe, base) is True
            unsafe = Path("/etc")
            assert is_safe_path(unsafe, base) is False
