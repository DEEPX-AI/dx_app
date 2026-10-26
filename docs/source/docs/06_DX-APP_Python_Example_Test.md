# DX-APP Python Example Tests

## Overview 

The Python Example project includes a comprehensive test framework built on `pytest`. It allows developers to validate script integrity, verify CLI arguments, and perform automated performance benchmarking across different models and hardware variants.  

**Key Features**  

- **Framework-Based:** Uses common base classes to ensure consistent testing logic across the entire repository.  
- **Layered Testing Strategy:** Provides a comprehensive validation path - **Unit → CLI → E2E**.  
- **Smart Mocking:** Combines high-speed software mocks (hardware-free) with real NPU hardware validation.  
- **Centralized Configuration:** Model metadata comes from `config/model_registry.json`. Test discovery lives in `tests/test_helpers/`.  
- **Automated Performance Tracking:** E2E tests automatically capture and aggregate NPU performance metrics (FPS, Latency).  

---

## Test Classification & Prerequisites

### Test Strategy Levels

**Quick Start Tests**  

The framework employs a layered approach to isolate issues effectively  

- **Unit Tests (`pytest unit/`):** Verifies core logic using mocks. Fast and ideal for CI/CD pipelines without NPU hardware  
- **CLI Tests (`-m cli`):** Ensures command-line arguments correctly trigger intended input modes (image/video/camera)  

**Advanced Test Selection**  

- **E2E (End-to-End) Tests (`-m e2e`):** High-fidelity tests using real `.dxnn` models and NPU hardware to capture actual performance  
  Video E2E tests run every stream-capable model except the W6 face detectors (`VIDEO_TOO_SLOW_MODELS`) and the anomaly-detection and super-resolution examples (listed in `IMAGE_ONLY_TASKS` in `tests/test_helpers/constants.py`). Super-resolution's stream path runs on a 6-frame low-resolution clip (`test_super_resolution_stream_e2e`).  

**Feature Tests (selection)**  

| File | Marker | Scope |
|------|--------|-------|
| `test_verify.py` | `verify` | `DXAPP_VERIFY` output of every downloaded model, sync and async (`*_cpp_postprocess` included): one non-empty record, per-frame file |
| `test_sync_async_parity.py` | `verify` | Async output equals sync frame by frame (120-frame video clip; `yolov5-s_640x640`, `yolopv2_384x640`, `superpoint_480x640` and one model per postprocessor family) |
| `test_signal_handling.py` | `signal_handling` | SIGINT graceful shutdown; SIGTERM in save mode finalizes the video and prints the summary |

### Environment Setup

**Configuration Reference**  

- `pytest.ini`: Defines the fixed custom markers, log formats, and global settings. `conftest.py` also registers one marker per model family and variant directory under `src/python_example/` at run time.  
- `conftest.py`: Contains shared fixtures, mock infrastructure, and setup/teardown utilities.  

**Interpreter for the examples (`DXAPP_TEST_PYTHON`)**  

The mocks in `conftest.py` exist only inside the pytest process. Most tests launch the real example scripts as subprocesses, and those import `dx_engine`, `dx_postprocess` and `cv2` for real. The interpreter for those subprocesses is chosen in this order:

1. `$DXAPP_TEST_PYTHON`, when set.
2. The dx-runtime venv, `dx-runtime/venv-dx-runtime/bin/python3`, when it exists.
3. The interpreter running pytest.

The pytest header prints the choice (`example python (DXAPP_TEST_PYTHON): ...`). The simplest setup is to run pytest itself from that venv:

```bash
VENV=../venv-dx-runtime                     # from dx_app/
$VENV/bin/python -m pip install -r requirements.txt -r tests/python_example/requirements.txt
$VENV/bin/python -m pytest tests/python_example
# another interpreter for the examples:
DXAPP_TEST_PYTHON=/path/to/python3 $VENV/bin/python -m pytest tests/python_example
```

Behind a TLS-intercepting proxy, pass the system CA bundle to pip (`--cert /etc/ssl/certs/ca-certificates.crt`); never disable verification.

Without `DISPLAY`/`WAYLAND_DISPLAY`, both suites set `QT_QPA_PLATFORM=offscreen` for the examples they start, so OpenCV windows do not abort on a headless machine. When an example subprocess times out, it gets SIGTERM first, then SIGKILL if it is still alive (`tests/test_helpers/proc.py`).

!!! note "Coverage Scope"

    Variant scripts are discovered from `src/python_example/<task>/<family>/<variant>/` by `tests/test_helpers/utils.py`. Adding a new source directory does not automatically guarantee full test coverage until registry metadata and the discovery helpers agree.

---

## Test Infrastructure

**Project Structure & Reference**  

The test files sit next to each other. They are not split into per-task directories.

```text   
tests/python_example/
├── README.md
├── conftest.py
├── pytest.ini
├── requirements.txt
├── test_cli_basic.py
├── test_cli_help.py
├── test_classification_parity.py
├── test_dump_tensors.py
├── test_e2e.py
├── test_e2e_camera_rtsp.py
├── test_multi_loop.py
├── test_preopt_contract.py
├── test_save_mode.py
├── test_signal_handling.py
├── test_sync_async_parity.py
├── test_verify.py
├── test_visualization.py
└── unit/                   # Processor, layout, and registry unit tests
```

### Shared Module (`tests/test_helpers/`)

All test files import shared constants and utilities from `tests/test_helpers/`:  

- `constants.py`: `TASK_IMAGE_MAP`, `MODEL_IMAGE_OVERRIDE`, path constants  
- `utils.py`: `setup_environment()`, `discover_python_scripts()`, `strip_variant_suffix()`  
- `proc.py`: subprocess runs with a timeout (SIGTERM first, then SIGKILL)  
- `pytest_support.py`: shared pytest options (`--loop`, `--camera-index`, `--rtsp-url`, `--stream-duration`)  
- `verify.py`: readers and comparers for `DXAPP_VERIFY` output  

### Asset Management

Assets required for testing are automatically managed by the setup scripts in the project root:  

- **Models**: Run `./setup_sample_models.sh` to fetch models for E2E validation.  
- **Data**: Run `./setup.sh` to ensure sample images and videos are placed in the expected directories.  

---

## Test Execution Guide

### Execution Methods

**Method A. Quick Start Guide**  

Navigate to the `tests/python_example/` directory to execute tests.  

```bash
# 1. Install testing dependencies
pip install -r requirements.txt

# 2. Run all available tests (Unit + CLI + E2E)
pytest

# 3. Skip the e2e-marked tests. The save_mode, dump_tensors, multi_loop, signal_handling,
#    verify and visualization tests still run on the NPU for every downloaded model;
#    for a hardware-free run use: pytest unit/ and pytest -m cli
pytest -m "not e2e"

# Specific test levels
pytest unit/            # Unit tests only
pytest -m cli           # CLI tests only
pytest -m e2e           # End-to-end tests only
```

**Method B. Manual Execution (Advanced Selection)**  

Filter tests by model family or variant to optimize development time.  

```bash
# By model family (substring of the test id, e.g. yolov7_640x640_sync)
pytest -k yolov7
pytest -k scrfd
pytest -k efficientnet

# By variant (marker)
pytest -m yolov7_640x640

# Combinations
pytest -m e2e -k "yolov7 or yolov8"
pytest -m cli -k yolov7
```

### Result Analysis

**E2E Hardware Benchmarking**  

The E2E suite functions as an **automated benchmarking tool** for hardware performance.  

- **Requirements:** Requires installed `dx_engine, dx_postprocess`, and relevant model assets. In practice, prepare assets with `./setup.sh` or `./setup_sample_models.sh` before E2E execution.  
- **Visual Output:** `E2E_DISPLAY=1 pytest -m e2e` drops `--no-display`. Image tests (sync and async) open a window and wait for q/ESC on every result; the video test skips async scripts in this mode.  

```bash
# Run E2E with display
E2E_DISPLAY=1 pytest -m e2e
```

!!! note "NOTE"  

    With `E2E_DISPLAY=1` the video E2E test runs sync scripts only, and every image test waits for a key press.  

- **Performance Reporting:** After E2E tests finish, a performance summary is printed to the console. Detailed logs, including FPS and Latency per model variant, are automatically saved to `tests/python_example/performance_reports/performance_report_{e2e|e2e_short}_YYYYMMDD_HHMMSS.csv`. The label is `e2e_short` when the run selects that marker, and `e2e` otherwise.  

---

## Repository Checks

The Python examples are covered by the same `bash scripts/ci_checks.sh` that checks the C++ tree without an NPU. See *Repository Checks* in [DX-APP C++ Example Tests](04_DX-APP_CPP_Example_Test.md).

---
