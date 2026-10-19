# DX-APP C++ Example Tests

## Overview

The project provides a comprehensive Python-based test suite located in `tests/cpp_example/`. These tests ensure that C++ binaries are functional, handle arguments correctly, produce correct visualizations, and meet performance benchmarks on the DEEPX NPU.

---

## Test Classification & Prerequisites

### Test Categories

The suite is organized into tiered categories based on execution speed and scope.

**CLI Tests (Fast)**  

- **Files**: `test_cli_help.py`, `test_cli_basic.py`  
- **Scope**: Validates `--help` options, `--version` flag, invalid argument handling, and no-argument behaviors for all executables.  

**E2E (End-to-End) Tests (Slow)**  

- **File**: `test_e2e.py`  
- **Scope**: Performs real inference on images and videos using `.dxnn` models to verify the full pipeline and NPU utilization.  
     : Auto-discovers all (executable, model) pairs from `bin/` and `assets/models/`.  
     : Assets: Uses real models from `assets/models/` and test data from `sample/img/` and `assets/videos/`.  
     : Parameters: Default loop count configurable via `--loop`.  
     : Timeouts: 100 seconds for image inference (300s for TTA models), 2000 seconds (about 33 minutes) for video inference.  
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
| `test_sync_async_parity.py` | `verify` | Async output equals sync frame by frame (video clip, DXRT_TASK_MAX_LOAD=40): every video-capable async runner, plus loops and an interrupt mid-video. The sweep leaves out the image-only tasks (super_resolution among them) and the registry rows marked `image_only`; `espcn-x4_17x17`, `realesrgan-x2_192x192`, `sfa3d_608x608` and `arcface_mobilefacenet_112x112` are compared in image mode instead. `yolopv2_384x640` runs at DXRT_TASK_MAX_LOAD=20 (`MAX_LOAD` in the file); other models that get fewer than 40 buffers run at 40 with dxrt's reduced count |
| `test_multi_loop.py` | `multi_loop` | `-l N` loop count behavior |
| `test_signal_handling.py` | `signal_handling` | SIGINT graceful shutdown; one Ctrl-C under `timeout` is one request; a later Ctrl-C ends a run stuck in its output (a FIFO) |

`test_graph_engine.py` (marker `graph`) runs the graph engine's `graph_engine_test` binary: it fails on a missing summary line, any failure, an OpenCV or GLib log line on stderr, a `Config file not found` line or any file left in its scratch `TMPDIR`; skipped hardware cases are reported as a pytest SKIP naming them.

### Environment Setup

**Test Requirements**  
Before running tests, ensure the environment is prepared.

For CLI Tests  

- Executables must be built in the `bin/` directory.  
- Run `./build.sh` from the project root if executables are missing.  

For E2E Tests (Additional)  

- **Models**: Run `./setup_sample_models.sh` to populate `assets/models/`. Pass `--all` to download every model non-interactively.  
- **Test Data**: Ensure images exist in `sample/img/` and videos in `assets/videos/` (Run `./setup_sample_videos.sh`).  
- **Libraries**: Shared libraries must be present in the `lib/` directory.  

**Available Markers (`pytest.ini`)**  

`cli`, `help`, `e2e`, `visualization`, `async_exec`, `sync_exec`, `save_mode`, `dump_tensors`, `verify`, `multi_loop`, `signal_handling`

---

## Test Infrastructure

### Shared Module (`tests/test_helpers/`)

All test files import shared constants and utilities from `tests/test_helpers/`:  

- `constants.py`: `TASK_IMAGE_MAP`, `MODEL_IMAGE_OVERRIDE`, `MULTI_MODEL_EXECUTABLES`, path constants  
- `utils.py`: `setup_environment()`, `discover_cpp_executables()`, `normalize_model_name()`  

### Asset Management

Assets required for testing are automatically managed by scripts within the `scripts/` directory:  

- **Models**: Use `setup_sample_models.sh` to download lightweight, test-specific models.  
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

# Run all tests (excluding slow E2E tests)
pytest -m "not e2e"

# Run only CLI tests (fast)
pytest test_cli_help.py test_cli_basic.py

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
pytest -m e2e -k "yolov7_async"     # Only yolov7_async

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
pytest test_e2e.py::test_stream_inference_e2e[yolov7_async]
pytest test_e2e.py::test_image_inference_e2e[scrfd_async]

# Multiple specific tests
pytest test_e2e.py::test_stream_inference_e2e[yolov7_async] \
            test_e2e.py::test_stream_inference_e2e[yolov8_async]
```

### Result Analysis

**Performance Reporting**  

After running E2E tests, a performance report is automatically generated to provide deep insights into the inference pipeline efficiency.  

- **Console Output**: A formatted table with real-time FPS metrics.  
- **CSV File**: A detailed log saved as `tests/cpp_example/performance_reports/performance_report_YYYYMMDD_HHMMSS.csv`.  

The report includes  

- **E2E FPS:** Overall pipeline throughput.  
- **Read FPS:** Speed of frame ingestion.  
- **Preprocess FPS:** Speed of image transformation (resizing, normalization).  
- **Inference FPS:** Pure NPU model execution speed.  
- **Postprocess FPS:** Speed of result parsing (NMS, coordinate scaling).  
- **Bottleneck Detection:** The slowest stage in the pipeline is automatically marked with an **asterisk (*)** for quick optimization targeting.  

**Test Coverage Summary**  

Example output: `./run_tc.sh --cpp --coverage`  

| **Category** | **Count** | **Status** |
|----|----|----|
| **CLI Tests** | ~1,293 | All binaries validated (help + basic) |
| **E2E Image Tests** | ~242 | sync + async, auto-discovered |
| **Visualization Tests** | ~247 | sync + async image verification |
| **Feature Tests** | ~22 | save_mode, dump_tensors, verify, multi_loop, signal_handling |

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
./build.sh --clean --coverage --type debug

# [Alternative] Build with relwithdebinfo for faster builds (less accurate coverage)
./build.sh --clean --coverage --type relwithdebinfo

# Verify coverage build (check for .gcno files)
ls build_x86_64/src/examples/*.gcno
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
./build.sh --clean --coverage --type debug && ./run_tc.sh --cpp --coverage

# Run C++ coverage (standalone)
./run_tc.sh --cpp --coverage
```

**Report Interpretation** (HTML Report & Filtering)  

After running tests with `--coverage`, several reports are generated in `tests/cpp_example/coverage/`

- **Console summary:** Line and branch coverage percentages.
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

## Maintenance & CI/CD

### Repository checks

`scripts/ci_checks.sh` is the single entry point for the repository checks that need no NPU; those needing the DX-RT headers SKIP where they are missing. Developers and CI jobs run the same command.

```bash
bash scripts/ci_checks.sh                  # every check, then a PASS / SKIP / FAIL summary
bash scripts/ci_checks.sh --list           # the check names, in order
bash scripts/ci_checks.sh --only guard-graph-boundary,python-compile
bash scripts/ci_checks.sh --require-dxrt   # a check skipped for want of the dxrt headers FAILS
```

| Check | What it runs | Needs |
|---|---|---|
| `guard-graph-boundary` | `scripts/check_graph_boundary.py`: the engine never names a concrete registry | Python |
| `guard-factory-uniqueness` | `scripts/check_factory_uniqueness.py`: no two factory headers declare the same fully qualified class (`dxapp::v_<variant>::<Class>`) | Python |
| `guard-model-registry` | `scripts/check_model_registry.py`: `config/model_registry.json` matches the factory tree | Python |
| `guard-variant-scope` | `scripts/generate_cpp_family_layout.py --variant-scope src/cpp_example --check`: every per-variant factory header is wrapped in its own `dxapp::v_<variant>` namespace, so two variants of one family linked into one program cannot silently share one class definition | Python |
| `codegen-strict` | `scripts/gen_model_registry.py --strict` into a temporary directory | Python |
| `codegen-check-docs` | `scripts/check_graph_models_doc.py`: `docs/graph_models.md` is not stale | Python |
| `header-odr` | `scripts/check_header_odr.sh`: two translation units including every shared header link; then two variants of one family linked into one program keep their own postprocessor (`scripts/check_variant_odr.sh`) | dxrt and OpenCV headers, g++ |
| `cxx14-headers` | `scripts/check_cxx14.sh`: the headers that own `g_interrupted()` compile as C++14 with `-Werror` | dxrt and OpenCV headers, g++ |
| `cross-compile` | `scripts/check_cross_compile.sh`: the graph engine and CLI compile for aarch64 | `aarch64-linux-gnu-g++`, dxrt and OpenCV headers |
| `python-compile` | `python -m compileall` over `src/python_example`, `src/bindings/python`, `scripts` and `tests` | Python |
| `tests-scripts` | `pytest tests/scripts` (hermetic, see `tests/README.md`) with `--known-failures tests/scripts/known_target_failures.txt` | pytest, PyYAML, requests (cmake and g++ for the tests that use them; the others skip) |

**Known TARGET failures.** `tests/scripts/known_target_failures.txt` lists the `tests/scripts` tests that are known to fail on the release base, one `<test id> | <reason>` per line. `tests-scripts` runs each of them as `xfail(strict=True)` (`tests/scripts/known_failures.py`): the check stays green while they fail, and turns red (XPASS(strict)) the day one passes, until its line is deleted. A malformed line, a test listed twice or a missing list file is a usage error (exit 4). A plain `pytest tests/scripts` reads no list and shows those tests failing. The list today holds only its header, which is accepted: `test_generate_build_bat.py::test_run_bat_executes_bare_relative_filename` now skips outside Windows (`skipif(os.name != "nt")`, it runs a `.bat` through `cmd.exe`), and the `test_setup_demo_models.py` tests take the expected count from `run_demo.py`'s `DEMOS`.

**SKIP versus FAIL.** A check whose prerequisite is missing on this machine prints `SKIP` with the reason, and the run still exits 0; only a `FAIL` makes it exit 1. `--require-dxrt` turns the missing dxrt headers of `header-odr`, `cxx14-headers` and `cross-compile` into a `FAIL`. A missing cross compiler is still a `SKIP`. `--only` runs the named checks (an unknown name exits 2).

**A machine without the dxrt headers.** Those three checks are skipped where the headers are missing. To see that run on a machine that has them:

```bash
DXRT_INCLUDE_DIR=/nonexistent bash scripts/ci_checks.sh
```

Other variables: `PYTHON` (the interpreter for the checks, default the active virtual environment's, else `python3`) and `DXAPP_CROSS_CXX`.

### Continuous Integration (CI) Integration

**Coverage Reports**  

The XML reports generated via the `--coverage` flag can be fed to a code-quality server (for example SonarQube) for static analysis and test coverage tracking, which helps identify untested logic paths during the development lifecycle.  

**Scheduled Regression Runs**  

Running the full E2E suite on a schedule (for example nightly) catches regressions caused by updates to NPU drivers, firmware, and the underlying runtime environment.  

---
