# Python Example Test Framework

Comprehensive test framework for Python Examples.

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run all tests (recommended)
pytest

# Fast tests only (skip E2E)
pytest -m "not e2e"

# Specific test levels
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m cli           # CLI tests only
pytest -m e2e           # End-to-end tests only
```


## Test Levels

### 1. Unit Tests
Mock-based tests for core functionality. Combines **base tests** and **group tests**

### 2. Integration Tests
Error handling and edge cases. Tests missing files, empty videos, keyboard interrupts, and successful inference flows with proper resource cleanup.

### 3. CLI Tests
Command-line argument validation. Tests help messages, missing required arguments, unrecognized options, and various input modes (image/video/camera/rtsp).

### 4. E2E Tests
Real model inference with actual `.dxnn` files and videos. Tests image and stream inference with real hardware, collecting performance metrics (FPS, latency).

**Requirements:**
- `dx_engine` module must be installed
- `dx_postprocess` module must be installed
- Actual model files (`.dxnn`) and test videos required

**Display Mode:**
- Set `E2E_DISPLAY=1` to enable visual output (cv2.imshow)
- Only works with **sync** variants (async variants skip display mode due to thread-safety)

```bash
# Run E2E with display
E2E_DISPLAY=1 pytest -m "e2e"
```

**Performance Report:**
- After E2E tests complete, performance metrics are displayed in the console and saved to `performance_reports/performance_report.csv`


## Test Markers

Run specific test subsets using markers:

```bash
# By model
pytest -m yolov7
pytest -m scrfd
pytest -m efficientnet

# By task type
pytest -m object_detection
pytest -m classification
pytest -m semantic_segmentation

# Combinations
pytest -m "unit and yolov7"
pytest -m "(yolov7 or yolov8) and not e2e"
```

## Project Structure

```
tests/python_example/
├── framework/              # Test framework core
│   ├── config.py           # Model configurations
│   ├── base_test.py        # Unit test base
│   ├── groups_test.py      # Variant group test
│   ├── integration_test.py # Integration template
│   ├── cli_test.py         # CLI template
│   └── e2e_test.py         # E2E template
│
├── object_detection/       # Object detection tests
├── classification/         # Classification tests
└── semantic_segmentation/  # Segmentation tests
```


## Key Features

- **Framework-Based**: All models use common test framework classes
- **Layered Testing**: Unit → Integration → CLI → E2E
- **Smart Mocking**: Fast mocked tests + real E2E validation
- **Centralized Config**: Single source of truth for model metadata
- **Auto Performance Tracking**: E2E tests collect FPS metrics automatically

## Documentation

- `pytest.ini` - pytest configuration and markers
- `conftest.py` - shared fixtures and setup
