## Overview 

The Python Example project includes a comprehensive test framework built on `pytest`. It allows developers to validate script integrity, verify CLI arguments, and perform automated performance benchmarking across different models and hardware variants.  

**Key Features**  

- **Framework-Based:** Uses common base classes to ensure consistent testing logic across the entire repository.  
- **Layered Testing Strategy:** Provides a comprehensive validation path - **Unit → Integration → CLI → E2E**.  
- **Smart Mocking:** Combines high-speed software mocks (hardware-free) with real NPU hardware validation.  
- **Centralized Configuration:** Model metadata is managed in a single source of truth (config.py) for simple maintenance.  
- **Automated Performance Tracking:** E2E tests automatically capture and aggregate NPU performance metrics (FPS, Latency).  

---

## Test Strategy Levels

The framework employs a layered approach to isolate issues effectively  

- (1) **Unit Tests (`-m unit`):** Verifies core logic using mocks. Fast and ideal for CI/CD pipelines without NPU hardware  
- (2) **Integration Tests (`-m integration`)** Validates error handling (e.g., missing files) and resource cleanup during interrupts  (`Ctrl+C`)  
- (3) **CLI Tests (`-m cli`):** Ensures command-line arguments correctly trigger intended input modes (image/video/camera)  
- (4) **E2E (End-to-End) Tests (`-m e2e`):** High-fidelity tests using real `.dxnn` models and NPU hardware to capture actual performance  

---

## Quick Start & Advanced Test

**Quick Start**  

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

**Advanced Test Selection**  

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

---

## E2E Hardware Benchmarking

The E2E suite functions as an **automated benchmarking tool** for hardware performance.  

- **Requirements:** Requires installed `dx_engine, dx_postprocess`, and relevant model assets.  
- **Visual Output:** Use `E2E_DISPLAY=1 pytest -m e2e` to enable UI output (available for sync variants).  

```bash
# Run E2E with display
E2E_DISPLAY=1 pytest -m e2e
```

!!! note "NOTE"  
    Display is only supported for `sync` variants due to thread-safety constraints.  

- **Performance Reporting:** After E2E tests finish, a performance summary is printed to the console. Detailed logs, including FPS and Latency per model variant, are automatically saved to: `tests/python_example/performance_reports/performance_report.csv`  

---

## Project Structure & Reference

The test suite mirrors the example structure for consistency  

```text   
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

**Configuration Reference**  

- `pytest.ini`: Defines custom markers, log formats, and global settings.  
- `conftest.py`: Contains shared fixtures and setup/teardown utilities.  

---
