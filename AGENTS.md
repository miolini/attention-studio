# AGENTS.md - AttentionStudio Development Guide

## Overview

AttentionStudio is a desktop application for visualizing, analyzing, and researching Large Language Models using Complete Replacement Models (CRM) methodology. It's a Python project using PyTorch, Transformers, PySide6, and pytest.

---

## Build, Lint, and Test Commands

### Installation

```bash
pip install -e ".[dev]"  # Install with dev dependencies
```

### Running Tests

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/unit/test_crm_trainer.py

# Run a single test function
pytest tests/unit/test_crm_trainer.py::test_transcoder_forward

# Run tests with coverage
pytest --cov=attention_studio --cov-report=html

# Run only unit tests (skip integration/slow)
pytest -m "not integration and not slow"

# Run only integration tests
pytest -m integration

# Run specific test marker
pytest -m slow
```

### Linting and Formatting

```bash
# Run ruff linter
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Run mypy type checker
mypy attention_studio/

# Run all checks (ruff + mypy)
ruff check . && mypy attention_studio/
```

### Building

```bash
# Build wheel with hatch
hatch build

# Build with specific target
hatch build wheel
```

---

## Code Style Guidelines

### General

- **Python Version**: 3.10+
- **Line Length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)

### Imports

- Use absolute imports: `from attention_studio.core.crm_trainer import Transcoder`
- Group imports in order: stdlib, third-party, local
- Sort alphabetically within groups
- Example:

```python
# stdlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# third-party
import torch
import torch.nn as nn
from PySide6.QtWidgets import QWidget

# local
from attention_studio.core.crm_trainer import Transcoder, TranscoderConfig
from attention_studio.ui.main_window import MainWindow
```

### Naming Conventions

- **Modules**: `snake_case` (e.g., `crm_trainer.py`)
- **Classes**: `PascalCase` (e.g., `ModelManager`, `TranscoderConfig`)
- **Functions/Methods**: `snake_case` (e.g., `load_model`, `get_layer_info`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_VRAM_GB`)
- **Private methods**: prefix with underscore (e.g., `_download_from_hub`)

### Type Annotations

- Use type hints for all function signatures
- Prefer `typing` module over builtins for complex types
- Use `Optional[X]` instead of `X | None`
- Example:

```python
def load_model(
    config: ModelConfig,
    progress: Optional[Callable[[float], None]] = None
) -> None:
    ...
```

### Data Classes

Use `@dataclass` for simple data containers:

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    path: Optional[Path] = None
    device: str = "cuda"
    dtype: str = "float16"
```

### Error Handling

- Use specific exception types
- Always catch with explicit exception variables
- Avoid bare `except:` clauses
- Example:

```python
try:
    result = json.loads(content)
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse JSON: {e}")
    return None

# For expected "no path" cases in graph algorithms
try:
    path = nx.shortest_path(graph, source, target)
except nx.NetworkXNoPath:
    return None
```

### Async Code

- Use `async`/`await` for I/O-bound operations
- Use `pytest-asyncio` for async tests
- Mark async tests with `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    model_manager = ModelManager(cache_dir="./cache")
    model = await model_manager.load_model("gpt2")
```

### Test Organization

- Place tests in `tests/` directory
- Follow the structure:
  ```
  tests/
  ├── unit/
  │   ├── test_crm_trainer.py
  │   ├── test_feature_extractor.py
  │   └── test_graph_builder.py
  ├── integration/
  │   ├── test_crm_pipeline.py
  │   └── test_export.py
  └── ui/
      └── test_main_window.py
  ```
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use fixtures from `conftest.py`

### Project Structure

```
attention_studio/
├── src/attention_studio/
│   ├── __main__.py
│   ├── core/                    # Core business logic
│   │   ├── model_manager.py
│   │   ├── crm_trainer.py
│   │   ├── feature_extractor.py
│   │   └── graph_builder.py
│   ├── agents/                  # LLM agents
│   │   ├── base_agent.py
│   │   ├── feature_agent.py
│   │   └── circuit_agent.py
│   ├── ui/                      # PySide6 UI
│   │   ├── main_window.py
│   │   └── widgets/
│   └── utils/                   # Utilities
│       ├── config.py
│       └── logging.py
└── tests/
    ├── unit/
    ├── integration/
    └── ui/
```

### Logging

- Use `loguru` for logging (already in dependencies)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Example:

```python
from loguru import logger

logger.info(f"Loading model: {config.name}")
logger.warning(f"Low memory available: {free_gb}GB")
```

### Configuration

- Use `pydantic` for configuration models
- Use `pydantic-settings` for environment-based config
- Store secrets in `.env` (never commit)

### UI Development (PySide6)

- Create Qt widgets, not modify existing
- Use `QApplication.instance()` for fixture
- Example fixture:

```python
@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])
```

### UI Thread Safety

- **NEVER block the UI thread** with long-running calculations (model loading, feature extraction, etc.)
- Use `_run_async()` helper for running inference in background threads
- Use `QThread` for background tasks - see `model_viz.py` for example pattern
- Use Qt Signals to communicate progress back to UI thread
- Always clean up threads properly - call `quit()` and `wait()` before starting new operations

**Training exceptions**: Training must run on main thread due to MPS stability issues. It automatically falls back to CPU when MPS is detected.

Example pattern:

```python
class OptimizationWorker(QObject):
    progress = Signal(dict)
    finished = Signal()

    def __init__(self, elements, connections, config):
        super().__init__()
        self._elements = elements
        self._connections = connections
        self._config = config

    def run(self):
        optimizer = ConstraintSatisfactionLayoutOptimizer(...)
        for positions in optimizer.optimize_incremental():
            self.progress.emit(positions)  # Update UI from background thread
        self.finished.emit()

# In your widget:
def start_optimization(self):
    self._thread = QThread()
    self._worker = OptimizationWorker(...)
    self._worker.moveToThread(self._thread)
    self._thread.started.connect(self._worker.run)
    self._worker.progress.connect(self._on_progress)  # Updates UI safely
    self._thread.start()
```

### Key Dependencies

- **Core**: torch>=2.0.0, transformers>=4.35.0
- **UI**: PySide6>=6.5.0, PyQtGraph>=0.13.0
- **Testing**: pytest>=7.4.0, pytest-asyncio>=0.21.0
- **Linting**: ruff>=0.1.0, mypy>=1.7.0

### CI/Development Scripts

- `scripts/build.sh` - Build application
- `scripts/test.sh` - Run test suite
- `scripts/release.sh` - Release process

---

## Notes

- This is a new project (PRD only, no code yet)
- Follow the patterns in PRD.md for implementation
- All paths in code should use `pathlib.Path`
- Use `device_map="auto"` for model loading (supports CUDA/MPS/CPU)

### Heavy Tasks and Threading

**IMPORTANT**: All heavy computational tasks must run off the UI thread to prevent freezing and crashes:

- **Model inference** (feature extraction, attribution graphs): Use `_run_async()` helper
- **Training**: Run synchronously on main thread (MPS/CPU stability issues with threading)
- **Model loading**: Already async via `_run_async()`

**Why this matters**:
- PyTorch MPS backend has threading issues - crashes when model operations run on background threads
- Training on MPS is unstable - automatically falls back to CPU
- UI blocking causes app to become unresponsive

**Current implementation**:
```python
# Use _run_async for inference (feature extraction, graphs)
def _on_extract_features(self):
    def extract():
        extractor = FeatureExtractor(...)
        features = extractor.extract_features(prompt, top_k=20)
        return features

    def on_done(features):
        # Update UI here
        self.output_table.setRowCount(len(features))

    def on_error(err):
        self.log_text.append(f"Error: {err}")

    self._run_async(extract, on_done, on_error)

# Training runs on main thread (synchronous) due to MPS stability
def _on_train(self):
    try:
        result = train()  # Runs on main thread
        on_done(result)
    except Exception as e:
        on_error(str(e))
```

**Device handling**:
- Model runs on MPS for fast inference
- Training automatically moves model to CPU (MPS has stability issues)
- Use `float32` when moving between devices to avoid dtype mismatches
