# DX-APP Python Example Tests

## Overview 

The Python Example project includes a comprehensive test framework built on `pytest`. It allows developers to validate script integrity, verify CLI arguments, and perform automated performance benchmarking across different models and hardware variants.  

**Key Features**  

- **Framework-Based:** Uses common base classes to ensure consistent testing logic across the entire repository.  
- **Layered Testing Strategy:** Provides a comprehensive validation path - **Unit → Integration → CLI → E2E**.  
- **Smart Mocking:** Combines high-speed software mocks (hardware-free) with real NPU hardware validation.  
- **Centralized Configuration:** Model metadata is managed in a single source of truth (config.py) for simple maintenance.  
- **Automated Performance Tracking:** E2E tests automatically capture and aggregate NPU performance metrics (FPS, Latency).  

---

## Test Classification & Prerequisites

### Test Strategy Levels

**Quick Start Tests**  

The framework employs a layered approach to isolate issues effectively  

- **Unit Tests (`-m unit`):** Verifies core logic using mocks. Fast and ideal for CI/CD pipelines without NPU hardware  
- **CLI Tests (`-m cli`):** Ensures command-line arguments correctly trigger intended input modes (image/video/camera)  

**Advanced Test Selection**  

- **Integration Tests (`-m integration`)** Validates error handling (e.g., missing files) and resource cleanup during interrupts  (`Ctrl+C`)  
- **E2E (End-to-End) Tests (`-m e2e`):** High-fidelity tests using real `.dxnn` models and NPU hardware to capture actual performance  

### Environment Setup

**Configuration Reference**  

- `pytest.ini`: Defines custom markers (50+), log formats, and global settings.  
- `conftest.py`: Contains shared fixtures, mock infrastructure, and setup/teardown utilities.  

!!! note "Coverage Scope"
    The Python test framework uses centralized model registration under `tests/python_example/framework/`. Adding a new source directory under `src/python_example/` does not automatically guarantee full test coverage until the corresponding test registration and mappings are updated.

---

## Test Infrastructure

**Project Structure & Reference**  

The test suite mirrors the example structure for consistency  

```text   
tests/python_example/
├── framework/              # Test framework core
│   ├── config.py           # Model configurations (60+ models)
│   ├── base_test.py        # Unit test base
│   ├── groups_test.py      # Variant group test
│   ├── integration_test.py # Integration template
│   ├── cli_test.py         # CLI template
│   ├── e2e_test.py         # E2E template
│   └── performance_collector.py  # Performance metrics
│
├── test_visualization.py   # Visualization tests (sync + async)
│                           #   Output → tests/test_visualization_result/python_example/{sync,async}/<task>/
│
├── object_detection/       # Object detection tests (25+ models)
├── classification/         # Classification tests
├── face_detection/         # Face detection tests
├── pose_estimation/        # Pose estimation tests
├── instance_segmentation/  # Instance segmentation tests
├── semantic_segmentation/  # Semantic segmentation tests
├── depth_estimation/       # Depth estimation tests
├── hand_landmark/          # Hand landmark tests
├── embedding/              # Embedding tests
├── obb_detection/          # OBB detection tests
├── image_denoising/        # Image denoising tests
├── image_enhancement/      # Image enhancement tests
├── super_resolution/       # Super resolution tests
└── ppu/                    # PPU model tests
```

### Shared Module (`tests/common/`)

All test files import shared constants and utilities from `tests/common/`:  

- `constants.py`: `TASK_IMAGE_MAP`, `MODEL_IMAGE_OVERRIDE`, path constants  
- `utils.py`: `setup_environment()`, `discover_python_scripts()`, `normalize_model_name()`  
### Asset Management

Assets required for testing are automatically managed by scripts within the `scripts/` directory:  

- **Models**: Run `./setup_sample_models.sh` to fetch lightweight models for E2E validation.  
- **Data**: Run `./setup.sh` to ensure sample images and videos are placed in the expected directories.  

---

## Test Execution Guide

### Execution Methods

**Method A. Quick Start Guide**  

Navigate to the `tests/python_example/` directory to execute tests.  

```bash
# 1. Install testing dependencies
pip install -r requirements.txt

# 2. Run all available tests (Unit + Integration + CLI + E2E)
pytest

# 3. Run only software-based tests (skips NPU hardware)
pytest -m "not e2e"

# Specific test levels
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m cli           # CLI tests only
pytest -m e2e           # End-to-end tests only
```

**Method B. Manual Execution (Advanced Selection)**  

Filter tests by model family or AI task to optimize development time.  

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

### Result Analysis

**E2E Hardware Benchmarking**  

The E2E suite functions as an **automated benchmarking tool** for hardware performance.  

- **Requirements:** Requires installed `dx_engine, dx_postprocess`, and relevant model assets. In practice, prepare assets with `./setup.sh` or `./setup_sample_models.sh` before E2E execution.  
- **Visual Output:** Use `E2E_DISPLAY=1 pytest -m e2e` to enable UI output (available for sync variants).  

```bash
# Run E2E with display
E2E_DISPLAY=1 pytest -m e2e
```

!!! note "NOTE"  
    Display is only supported for `sync` variants due to thread-safety constraints.  

- **Performance Reporting:** After E2E tests finish, a performance summary is printed to the console. Detailed logs, including FPS and Latency per model variant, are automatically saved to: `tests/python_example/performance_reports/performance_report.csv`  

---

## Maintenance & CI/CD

### Continuous Integration (CI) Integration

- **Automated Regression**: Python tests are integrated into the CI pipeline to catch breaking changes in post-processing logic or CLI interfaces.  
- **Performance Baseline**: Nightly E2E runs compare current FPS/Latency against historical baselines to detect performance regressions in the NPU software stack.  
- **Reporting**: Test results are exported in JUnit XML format for integration with CI dashboards (e.g., Jenkins, GitHub Actions).  

---
