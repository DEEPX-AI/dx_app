# DX-APP C++ Example Tests

## Overview

The project provides a comprehensive Python-based test suite located in `tests/cpp_example/`. These tests ensure that C++ binaries are functional, handle arguments correctly, produce correct visualizations, and report performance (FPS per pipeline stage) on the DEEPX NPU.

---

## Test Classification & Prerequisites

### Test Categories

The suite is organized into tiered categories based on execution speed and scope.

**CLI Tests (Fast)**  

- **Files**: `test_cli_help.py`, `test_cli_basic.py`  
- **Scope**: Validates `--help` output, invalid argument handling, and no-argument behaviors for all executables.  

**E2E (End-to-End) Tests (Slow)**  

- **File**: `test_e2e.py`  
- **Scope**: Performs real inference on images and videos using `.dxnn` models to verify the full pipeline and NPU utilization.  
     : Auto-discovers all (executable, model) pairs from `bin/` and `assets/models/`.  
     : Assets: Uses real models from `assets/models/` and test data from `sample/img/` and `assets/videos/`.  
     : Parameters: Default loop count configurable via `--loop`.  
     : Timeouts: no time limit for image inference except TTA models (300 seconds); models in `E2E_HEAVY_MODELS` (`tests/test_helpers/constants.py`) run at most 2 loops. Video inference: 2000 seconds (about 33 minutes).  
     : Video E2E and `--save` video tests run every stream-capable model except the W6 face detectors (`VIDEO_TOO_SLOW_MODELS` in `tests/test_helpers/constants.py`). Super-resolution's stream path runs on a 6-frame low-resolution clip (`test_super_resolution_stream_e2e`).  

**Specialized Tests**  

Visualization Tests  

- **File**: `test_visualization.py`  
- **Scope**: Runs all sync + async binaries with `DXAPP_SAVE_IMAGE` and verifies image output is produced.  
     : Output directory: `tests/test_visualization_result/cpp_example/{sync,async}/<task>/`  
     : Can also be run standalone: `python test_visualization.py`  

Feature Tests  

| File | Marker | Scope |
|------|--------|-------|
| `test_save_mode.py` | `save_mode` | `--save` / `--save-dir` output and run directory creation |
| `test_dump_tensors.py` | `dump_tensors` | `--dump-tensors` tensor file generation |
| `test_verify.py` | `verify` | `DXAPP_VERIFY` output of every downloaded model, sync and async: one non-empty record, per-frame file |
| `test_verify_dumps.py` | `verify` | Runtime checks of past `DXAPP_VERIFY` regressions (async classification/detection, restoration and tiled SR, 3D detection, YOLOPv2 masks) |
| `test_sync_async_parity.py` | `verify` | Async output equals sync frame by frame (video clip, at dxrt's default I/O buffer count): every video-capable async runner, plus loops and an interrupt mid-video. The sweep leaves out the image-only tasks (super_resolution among them) and the registry rows marked `image_only`; `espcn-x4_17x17`, `realesrgan-x2_192x192`, `sfa3d_608x608` and `arcface_mobilefacenet_112x112` are compared in image mode instead. |
| `test_multi_loop.py` | `multi_loop` | `-l N` loop count behavior |
| `test_signal_handling.py` | `signal_handling` | SIGINT graceful shutdown; one Ctrl-C under `timeout` is one request; a later Ctrl-C ends a run stuck in its output (a FIFO) |

`test_graph_engine.py` (marker `graph`) runs the graph engine's `graph_engine_test` binary: it fails on a missing summary line, any failure, an OpenCV or GLib log line on stderr, a `Config file not found` line or any file left in its scratch `TMPDIR`; skipped hardware cases are reported as a pytest SKIP naming them.

Other test files:

| File | Marker | Scope |
|------|--------|-------|
| `test_graph_cli.py` | `graph`, `graph_e2e`, `graph_parity` | Multi-model graph CLI. `graph`: no NPU, `--check` and `--list-models`. `graph_e2e`: every sample graph end to end on hardware. `graph_parity`: sync and async executors give byte-identical `--report` JSON |
| `test_graph_python.py` | `graph`, `graph_e2e`, `graph_parity` | The `dx_graph` Python binding of the graph engine (skipped when `dx_graph` is not built) |
| `test_graph_export.py` | `contract`, `graph_export`, `graph_e2e` | A graph exported with `scripts/export_graph_project.py`, built and run outside the repository (`graph_export` needs no NPU) |
| `test_e2e_camera_rtsp.py` | `e2e_camera`, `e2e_rtsp` | Camera (`-c`) and RTSP (`-r`) input: each executable runs for `--stream-duration` seconds, then gets SIGINT; passes on exit 0, frames > 0 and FPS > 0 |
| `test_headless_display.py` | `e2e`, `e2e_image` | An example started without a display server and without `--no-display` runs headless instead of crashing |
| `test_download_hints.py` | `contract` | Recovery hints name `scripts/download_models.py`, never `./setup.sh --models` |
| `test_common_unit.py` | — | Runs the `common_unit_test` binary (plain-assert tests of `common/`) |
| `test_harness_helpers.py` | `cli`, `help` | Test harness helpers: executable discovery, headless display, bounded subprocess runs |
| `test_preprocessor_scale_regression.py` | `golden`, `contract` | Golden checks of the grayscale resize preprocessor's per-axis scale (DnCNN, ESPCN, SuperPoint keypoints) |

### Environment Setup

**Test Requirements**  
Before running tests, ensure the environment is prepared.

For CLI Tests  

- Executables must be built in the `bin/` directory.  
- Run `./build.sh --all` from the project root to test every executable. The default `./build.sh` builds only the `run_demo` targets, so only those are tested; use `--target <name>` to build a specific one.  

For E2E Tests (Additional)  

- **Models**: Run `./setup_sample_models.sh` to populate `assets/models/`. Pass `--all` to download every model non-interactively.  
- **Test Data**: Ensure images exist in `sample/img/` and videos in `assets/videos/` (Run `./setup_sample_videos.sh`).  

**Available Markers (`pytest.ini`)**  

`cli`, `help`, `e2e`, `e2e_image`, `e2e_stream`, `e2e_camera`, `e2e_rtsp`, `e2e_short`, `visualization`, `async_exec`, `sync_exec`, `save_mode`, `dump_tensors`, `verify`, `multi_loop`, `signal_handling`, `contract`, `golden`, `graph`, `graph_e2e`, `graph_parity`, `graph_export`

---

## Test Infrastructure

### Shared Module (`tests/test_helpers/`)

All test files import shared constants and utilities from `tests/test_helpers/`:  

- `constants.py`: `TASK_IMAGE_MAP`, `MODEL_IMAGE_OVERRIDE`, `MULTI_MODEL_EXECUTABLES`, path constants  
- `utils.py`: `setup_environment()`, `discover_cpp_executables()`, `strip_variant_suffix()`, `dxnn_for_exe()`  
- `platform_paths.py`: per-OS `bin/` directory and `.exe` names  
- `proc.py`: subprocess runs with a timeout (SIGTERM first, then SIGKILL)  
- `pytest_support.py`: shared pytest options (`--loop`, `--camera-index`, `--rtsp-url`, `--stream-duration`)  
- `verify.py`: readers and comparers for `DXAPP_VERIFY` output  

### Asset Management

Assets required for testing are automatically managed by the setup scripts in the project root:  

- **Models**: Use `setup_sample_models.sh` to download models.  
- **Videos**: Use `setup_sample_videos.sh` to acquire sample video files required for E2E (End-to-End) pipeline verification.  

---

## Test Execution Guide

### Execution Methods

**Method A. Unified Test Runner (Recommended)**  

The `run_tc.sh` script provides a high-level interface for running standardized test suites directly from the project root.

```bash
cd ../../  # Go to project root (dx_app/)

# Run only C++ tests (CLI + E2E stream)
./run_tc.sh --cpp

# Run only C++ CLI tests (fast)
./run_tc.sh --cpp --cli

# Run only C++ E2E stream tests (all models)
./run_tc.sh --cpp --e2e

# Run only C++ E2E image tests (faster, skips stream)
./run_tc.sh --cpp --e2e-quick

# Run only C++ E2E image + stream tests for representative models (one per task, `E2E_SHORT_MODELS`)
./run_tc.sh --cpp --e2e-short

# Run C++ tests with code coverage (standalone)
./run_tc.sh --cpp --coverage

# Show all available options
./run_tc.sh --help
```

!!! note "TIP"
    Use `--e2e-quick` during development. It takes ~2–3 minutes, compared to 8–10 minutes for a full test.
    Use `--coverage` for coverage analysis — it cannot be combined with --cli, --e2e, etc.

**Method B. Manual Execution via Pytest**  

For granular control, run `pytest` directly from `tests/cpp_example/`.

Basic Usage  

```bash
# Install requirements
pip install -r requirements.txt

# Run every test not marked e2e (feature, visualization, verify and graph tests still run real binaries and can take long)
pytest -m "not e2e"

# Run only CLI tests (fast)
pytest -m cli

# Run only E2E tests (slow, requires models and test data)
pytest test_e2e.py -v

# Run all tests including E2E with verbose output
pytest -v -s
```

Advanced Filtering  

```bash
cd tests/cpp_example

# Test specific model (all variants)
pytest -m e2e -k "yolov7"           # All yolov7 variants
pytest -m e2e -k "yolov7_640x640_async"  # Only that async variant

# Test multiple models
pytest -m e2e -k "yolov7 or yolov8"
pytest -m e2e -k "scrfd or yolov5"

# Exclude variants
pytest -m e2e -k "yolov7 and not async"  # Only yolov7 sync variants
pytest -m e2e -k "yolov5 and not ppu"    # yolov5 without ppu variants

# Combine with markers
pytest -m "e2e and async_exec" -k "yolov7"  # Only async yolov7 tests
pytest -m "e2e and sync_exec" -k "yolov5"   # Only sync yolov5 tests

# Run specific test function
pytest test_e2e.py::test_stream_inference_e2e[yolov7_640x640_async]
pytest test_e2e.py::test_image_inference_e2e[scrfd-500m_640x640_async]

# Multiple specific tests
pytest test_e2e.py::test_stream_inference_e2e[yolov7_640x640_async] \
            test_e2e.py::test_stream_inference_e2e[yolov8-n_640x640_async]
```

### Result Analysis

**Performance Reporting**  

After running the E2E stream (video) tests, a performance report is automatically generated to provide deep insights into the inference pipeline efficiency.  

- **Console Output**: A formatted table with real-time FPS metrics.  
- **CSV File**: A detailed log saved as `tests/cpp_example/performance_reports/performance_report_YYYYMMDD_HHMMSS.csv`.  

The report includes  

- **E2E FPS:** Overall pipeline throughput.  
- **Read FPS:** Speed of frame ingestion.  
- **Preprocess FPS:** Speed of image transformation (resizing, normalization).  
- **Inference FPS:** Pure NPU model execution speed.  
- **Postprocess FPS:** Speed of result parsing (NMS, coordinate scaling).  
- **Bottleneck Detection:** For async variants, the slowest of the Read / Preprocess / Inference / Postprocess stages is marked with an **asterisk (*)** for quick optimization targeting.  

**Test Coverage Summary**  

`./run_tc.sh --cpp --coverage` runs one pytest pass, `-m "not e2e_stream and not visualization and not multi_loop and not signal_handling" --coverage`. It covers the CLI tests, E2E image tests and the save_mode, dump_tensors, verify, contract, golden and graph tests. Stream E2E, visualization, multi-loop and signal-handling tests are not included. Test counts depend on the executables in `bin/` (`./build.sh --all` builds a sync and an async binary per variant, about 1,000) and on the downloaded models.

---

## Advanced Analysis: Code Coverage

Code coverage measures how much of the source code is exercised during testing. This is essential for ensuring the robustness of the NPU inference pipeline.

### Coverage Analysis Strategy

To generate code coverage reports, you need to build the project with instrumentation and install the following tools

- **Step 1.** Build executables with coverage instrumentation.  
- **Step 2.** Install coverage tools (`gcovr` is recommended; `lcov` is supported as a fallback).  

```bash
# Install gcovr (recommended - supports XML, HTML, JSON)
sudo apt-get install gcovr -y

# OR install lcov (HTML only)
sudo apt-get install lcov -y

# [Recommended] Build with debug mode for the most reliable coverage results
cd ../../  # Go to project root (dx_app/) from tests/cpp_example/
./build.sh --clean --coverage --type debug --all 

# [Alternative] Build with relwithdebinfo for faster builds (less accurate coverage)
./build.sh --clean --coverage --type relwithdebinfo --all

# Verify coverage build (check for .gcno files)
find build_x86_64 -name "*.gcno" | head
```

**Build & Environment**  

- `relwithdebinfo`: Recommended for development (1.5–2x slower, optimized with debug info)
- `debug`: Most detailed coverage (3–5x slower, no optimization, full symbol info)

### Report Generation

Trigger coverage analysis using the following commands

**Execution Flow** (Manual vs Unified)  

Manual Execution (from `tests/cpp_example/`)  

```bash
# Quick E2E with coverage (image tests only, 2-3 minutes)
pytest -m e2e -k "test_image_inference_e2e" --coverage -v

# Full E2E tests with coverage (8-10 minutes)
pytest -m e2e --coverage -v

# Run specific models with coverage
pytest -m e2e -k "yolov7" --coverage -v
```

Unified Test Runner (from project root)  

```bash
# [Recommended] Build with debug mode, then run coverage
./build.sh --clean --coverage --type debug --all && ./run_tc.sh --cpp --coverage

# Run C++ coverage (standalone)
./run_tc.sh --cpp --coverage
```

**Report Interpretation** (HTML Report & Filtering)  

After running tests with `--coverage`, several reports are generated in `tests/cpp_example/coverage/`

- **Console summary:** line coverage (gcovr); line, function and branch coverage with the lcov fallback.
- **HTML report:** Located at `html/index.html`. Shows line-by-line visualization with uncovered code in **red**.
- **XML (Cobertura) & JSON:** Timestamped for CI/CD integration.

View the HTML Report  

```bash
# Open in your browser
firefox tests/cpp_example/coverage/html/index.html
# or
xdg-open tests/cpp_example/coverage/html/index.html
```

Report Features  

- Overall and file-by-file coverage statistics.  
- Line-by-line visualization with **uncovered code highlighted in red**.  
- Branch coverage analysis.  

Coverage Filtering Rules  

- **Included:** All source files within the `src/` directory.  
- **Excluded:** System headers (`/usr/*`), `third_party/*`, `extern/*`, and the `tests/` directory itself.

---

## Repository Checks

If you modify the examples or add a model, `scripts/ci_checks.sh` checks the repository's consistency (model registry, factory layout, generated sources, Python syntax) without an NPU.

```bash
bash scripts/ci_checks.sh          # run every check, then print a PASS / SKIP / FAIL summary
bash scripts/ci_checks.sh --list   # list the check names
```

A check whose prerequisite is missing on this machine (for example the DX-RT headers or a cross compiler) prints `SKIP` with the reason, and the run still exits 0. Only a `FAIL` makes it exit 1.

---
