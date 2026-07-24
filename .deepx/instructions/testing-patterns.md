# Testing Patterns for dx_app

Testing framework and patterns for validating dx_app inference applications.

## Test Framework

dx_app uses **pytest** as its test runner, with custom fixtures and utilities
for NPU-dependent testing.

### Running Tests

```bash
# Run all tests
./run_tc.sh

# Run specific test module
python -m pytest tests/python_example/ -v

# Run specific test by name
python -m pytest tests/python_example/ -k "test_yolov8n" -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### run_tc.sh

The `run_tc.sh` script is the primary test entry point. It:
1. Sets up the Python environment
2. Discovers and runs all test cases
3. Reports results with pass/fail summary

## Test Directory Structure

```
tests/
    CMakeLists.txt
    common/              # Shared test utilities
    cpp_example/         # C++ test cases
    python_example/      # Python test cases
    save_all_visualizations.py  # Visual regression helper
    src/                 # Additional test sources
```

## DXAPP_VERIFY Environment Variable

Setting `DXAPP_VERIFY=1` enables numerical verification output during inference.
The runner dumps a JSON file containing detection/classification results with
numerical coordinates and scores, which can be compared against golden references.

```bash
# Run with verification output
DXAPP_VERIFY=1 python yolov8n_sync.py --model model.dxnn --image test.jpg --no-display
```

The verify JSON contains:
```json
{
  "model": "yolov8n",
  "task": "object_detection",
  "image": "test.jpg",
  "image_size": [480, 640],
  "results": [
    {"class_id": 0, "score": 0.89, "box": [100, 50, 300, 400]},
    ...
  ]
}
```

## Test Fixtures

### NPU Availability

Tests that require NPU hardware should be decorated to skip when unavailable:

```python
import pytest
import subprocess


def is_npu_available():
    """Check if DEEPX NPU is accessible."""
    try:
        result = subprocess.run(
            ["dxrt-cli", "-s"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_npu = pytest.mark.skipif(
    not is_npu_available(),
    reason="DEEPX NPU not available"
)
```

Usage:
```python
@requires_npu
def test_yolov8n_sync_inference():
    """Test YOLOv8n sync inference with NPU."""
    ...
```

### Model Availability

Tests should also check that the model file exists:

```python
import os

def model_path(model_name):
    """Resolve model path from name."""
    # Check common model directories
    candidates = [
        f"models/{model_name}.dxnn",
        f"sample/{model_name}.dxnn",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


requires_model = lambda name: pytest.mark.skipif(
    model_path(name) is None,
    reason=f"Model {name}.dxnn not found"
)
```

### Test Image/Video Fixtures

```python
@pytest.fixture
def sample_image(tmp_path):
    """Create a synthetic test image."""
    import numpy as np
    import cv2
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    path = str(tmp_path / "test.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def sample_video(tmp_path):
    """Create a short synthetic test video."""
    import numpy as np
    import cv2
    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    for _ in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
```

## Test Categories

### 1. Factory Tests

Verify that factories implement all required methods and return correct types:

```python
def test_factory_implements_interface():
    factory = Yolov8Factory()
    assert hasattr(factory, 'create_preprocessor')
    assert hasattr(factory, 'create_postprocessor')
    assert hasattr(factory, 'create_visualizer')
    assert hasattr(factory, 'get_model_name')
    assert hasattr(factory, 'get_task_type')

def test_factory_model_name():
    factory = Yolov8Factory()
    assert factory.get_model_name() == "yolov8n"
    assert factory.get_task_type() == "object_detection"

def test_factory_load_config():
    factory = Yolov8Factory()
    factory.load_config({"score_threshold": 0.5})
    assert factory.config["score_threshold"] == 0.5
```

### 2. Preprocessor Tests

Verify preprocessing produces correct output shapes and ranges:

```python
def test_preprocessor_output_shape():
    factory = Yolov8Factory()
    pre = factory.create_preprocessor(640, 640)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor, ctx = pre.process(img)
    assert tensor.shape[-2:] == (640, 640)
```

### 3. Postprocessor Tests

Verify postprocessing handles empty and normal outputs:

```python
def test_postprocessor_empty_output():
    factory = Yolov8Factory()
    post = factory.create_postprocessor(640, 640)
    # Simulate empty model output
    outputs = [np.zeros((1, 84, 8400), dtype=np.float32)]
    results = post.process(outputs, None)
    assert isinstance(results, list)
```

### 4. Visualizer Tests

Verify visualization doesn't crash and returns valid images:

```python
def test_visualizer_no_results():
    factory = Yolov8Factory()
    vis = factory.create_visualizer()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    output = vis.visualize(img, [])
    assert output.shape == img.shape
```

### 5. Integration Tests (NPU Required)

Full end-to-end inference tests:

```python
@requires_npu
@requires_model("yolov8n")
def test_yolov8n_sync_e2e(sample_image):
    """End-to-end sync inference test."""
    import subprocess
    result = subprocess.run(
        ["python", "src/python_example/object_detection/yolov8n/yolov8n_sync.py",
         "--model", model_path("yolov8n"),
         "--image", sample_image,
         "--no-display"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0
    assert "[INFO] Starting inference" in result.stdout
```

### 6. Config Tests

Verify config.json files are valid:

```python
import json
import glob

def test_all_configs_valid():
    """All config.json files must be valid JSON."""
    configs = glob.glob("src/python_example/**/config.json", recursive=True)
    for path in configs:
        with open(path) as f:
            config = json.load(f)
        assert isinstance(config, dict), f"Invalid config: {path}"
```

### 7. Model Registry Tests

Verify model_registry.json integrity:

```python
def test_model_registry_schema():
    with open("config/model_registry.json") as f:
        models = json.load(f)
    assert isinstance(models, list)
    for m in models:
        assert "model_name" in m
        assert "dxnn_file" in m
        assert "add_model_task" in m
        assert "input_width" in m
        assert "input_height" in m
```

## 5-Level Validation Pyramid

```
Level 5:  Integration (NPU required, full pipeline)
Level 4:  Smoke (quick inference, sanity check)
Level 3:  Component (pre/post/viz individually)
Level 2:  Config (JSON validity, schema)
Level 1:  Static (imports, factory interface)
```

Run levels 1-3 without NPU hardware. Levels 4-5 require NPU + models.

## Common Test Failures and Fixes

| Failure | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: common` | sys.path not set | Add python_example/ to PYTHONPATH |
| `ModuleNotFoundError: dx_engine` | DX-RT not installed | Install dx_engine or skip test |
| `ModuleNotFoundError: dx_postprocess` | Bindings not built | Run `./build.sh` first |
| `FileNotFoundError: .dxnn` | Model not downloaded | Run `./setup.sh` |
| `RuntimeError: NPU not found` | No NPU hardware | Use `@requires_npu` skip |
| `TypeError: abstract class` | Factory incomplete | Implement all 5 methods |
