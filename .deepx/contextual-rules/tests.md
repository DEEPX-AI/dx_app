---
glob: "tests/**"
description: Rules for test files in dx_app.
---

# Test Rules

## pytest Framework

All tests use pytest. Run from dx_app root:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_detection.py

# Run tests matching a pattern
pytest tests/ -k "yolov8"
```

## Markers

Custom pytest markers for dx_app:

```python
import pytest

@pytest.mark.npu
def test_inference_on_npu():
    """Requires NPU hardware. Skipped if no device found."""
    ...

@pytest.mark.slow
def test_full_video_pipeline():
    """Takes > 30 seconds. Skipped in CI fast mode."""
    ...

@pytest.mark.model(name="yolov8n")
def test_yolov8n_detection():
    """Requires specific model to be downloaded."""
    ...
```

### NPU Skip Logic

Tests marked with `@pytest.mark.npu` are automatically skipped when no NPU device
is available:

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "npu: requires NPU hardware")
    config.addinivalue_line("markers", "slow: takes > 30 seconds")
    config.addinivalue_line("markers", "model: requires specific model")

def pytest_collection_modifyitems(items):
    if not _npu_available():
        skip_npu = pytest.mark.skip(reason="No NPU device found")
        for item in items:
            if "npu" in item.keywords:
                item.add_marker(skip_npu)

def _npu_available():
    try:
        import subprocess
        result = subprocess.run(
            ["dxrt-cli", "-s"],
            capture_output=True, text=True, timeout=5
        )
        return "ready" in result.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
```

## Shared Fixtures

Common test fixtures in `tests/conftest.py`:

```python
import pytest
import json
from pathlib import Path

@pytest.fixture
def model_registry():
    """Load model_registry.json."""
    registry_path = Path(__file__).parent.parent / "config" / "model_registry.json"
    with open(registry_path) as f:
        return json.load(f)

@pytest.fixture
def sample_image():
    """Provide a sample test image as numpy array."""
    import numpy as np
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_frame_640():
    """Provide a 640x640 preprocessed frame."""
    import numpy as np
    return np.random.rand(1, 640, 640, 3).astype(np.float32)

@pytest.fixture
def detection_config():
    """Standard detection config."""
    return {
        "score_threshold": 0.25,
        "nms_threshold": 0.45
    }
```

## DXAPP_VERIFY Environment Variable

The `DXAPP_VERIFY` environment variable controls validation strictness:

```bash
# Run tests with strict validation
DXAPP_VERIFY=strict pytest tests/

# Run tests with relaxed validation (skip model file checks)
DXAPP_VERIFY=relaxed pytest tests/

# Default: standard validation
pytest tests/
```

```python
import os

verify_level = os.environ.get("DXAPP_VERIFY", "standard")

if verify_level == "strict":
    # Check model files exist, verify tensor shapes, etc.
    assert model_path.exists(), f"Model file not found: {model_path}"
elif verify_level == "relaxed":
    # Skip expensive checks
    pytest.skip("Relaxed mode: skipping model file verification")
```

## Test Organization

```
tests/
    conftest.py                  # Shared fixtures, markers, NPU detection
    test_registry.py             # model_registry.json integrity tests
    test_detection.py            # Detection model tests
    test_classification.py       # Classification model tests
    test_segmentation.py         # Segmentation model tests
    test_pose.py                 # Pose estimation tests
    test_face.py                 # Face detection tests
    test_factory.py              # IFactory implementation tests
    test_runner.py               # SyncRunner/AsyncRunner tests
    test_postprocess.py          # dx_postprocess binding tests
    test_cli.py                  # parse_common_args() tests
    test_config.py               # config.json loading tests
```

## Test Naming

```python
# Pattern: test_<what>_<condition>_<expected>
def test_yolov8n_detection_returns_boxes():
    ...

def test_factory_missing_method_raises_typeerror():
    ...

def test_parse_args_default_threshold_is_025():
    ...
```

## Prohibited Patterns

- No `unittest.TestCase` (use pytest functions)
- No `print()` for test output (use pytest's `capsys` or `capfd`)
- No hardcoded absolute paths
- No tests that modify global state without cleanup
- No sleep-based synchronization (use proper fixtures/mocks)
