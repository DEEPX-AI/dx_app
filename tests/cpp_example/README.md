## Overview

The project provides a comprehensive Python-based test suite located in `tests/cpp_example/`. These tests ensure that C++ binaries are functional, handle arguments correctly, and meet performance benchmarks on the DEEPX NPU.  

---

## Test Categories

The suite is categorized into two main tiers based on execution speed and depth.  

**CLI Tests (Fast)**  

- **Files**: `test_cli_help.py, test_cli_basic.py`  
- **Scope**: Validates `--help` options, invalid argument handling, and no-argument behaviors for all 48 executables.  

**E2E(End-to-End) Tests (Slow)**  

- **File**: `test_e2e.py`  
- **Scope**: Performs real inference on images and videos using `.dxnn` models to verify the full pipeline and NPU utilization.  
     : Coverage: Only tests 36 executables that support the `--no-display` option.  
     : Assets: Uses real models from `assets/models/` and test data from `sample/img/` and `assets/videos/`.  
     : Parameters: Default loop count of 50 iterations.  
     : Timeouts: 100 seconds for image inference, 10 minutes for video inference.  

---

## Prerequisites

Before running tests, ensure the environment is prepared.  

**For CLI Tests**  

- Executables must be built in the `bin/` directory.  
- Run `./build.sh` from the project root if executables are missing.  

**For E2E Tests (Additional)**  

- **Models**: Run `./setup_sample_models.sh` to populate `assets/models/`.  
- **Test Data**: Ensure images exist in `sample/img/` and videos in `assets/videos/` (Run `./setup_sample_videos.sh`)`.  
- **Libraries**: Shared libraries must be present in the `lib/` directory.  

---

## Running Tests 

### Execution Methods (Standard)  

**Method A. Unified Test Runner (Recommended)**  

The `run_tc.sh` script provides a high-level interface for running standardized test suites directly from the project root.  

```bash
cd ../../  # Go to project root (dx_app/)

# Run only C++ tests
./run_tc.sh --cpp

# Run only C++ CLI tests (fast)
./run_tc.sh --cpp --cli

# Run only C++ E2E tests
./run_tc.sh --cpp --e2e

# Run only C++ E2E image tests (faster, skips video)
./run_tc.sh --cpp --e2e-quick

# Run C++ E2E tests with code coverage
./run_tc.sh --cpp --e2e --coverage

# Run quick E2E with coverage (recommended for development)
./run_tc.sh --cpp --e2e-quick --coverage

# Show all available options
./run_tc.sh --help
```

!!! note "TIP"  
    Use `--e2e-quick` during development. It takes ~2–3 minutes, compared to 8–10 minutes for a full test.  

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
pytest test_e2e.py::test_video_inference_e2e[yolov7_async]
pytest test_e2e.py::test_image_inference_e2e[scrfd_async]

# Multiple specific tests
pytest test_e2e.py::test_video_inference_e2e[yolov7_async] \
            test_e2e.py::test_video_inference_e2e[yolov8_async]
```

### Performance Reporting

After running E2E tests, a performance report is automatically generated to provide deep insights into the inference pipeline efficiency.  

- **Console Output**: A formatted table with real-time FPS metrics.  
- **CSV File**: A detailed log saved as `performance_report_YYYYMMDD_HHMMSS.csv` in `tests/cpp_example/`.  

The report includes  

- **E2E FPS:** Overall pipeline throughput.  
- **Read FPS:** Speed of frame ingestion.  
- **Preprocess FPS:** Speed of image transformation (resizing, normalization).  
- **Inference FPS:** Pure NPU model execution speed.  
- **Postprocess FPS:** Speed of result parsing (NMS, coordinate scaling).  
- **Bottleneck Detection:** The slowest stage in the pipeline is automatically marked with an **asterisk (*)** for quick optimization targeting.  

---

## Code Coverage Analysis

Code coverage measures how much of the source code is exercised during testing. This is essential for ensuring the robustness of the NPU inference pipeline.  

### Prerequisites  

To generate code coverage reports, you need to build the project with instrumentation and install the following tools  

- **Step 1.** Build executables with coverage instrumentation.  
- **Step 2.** Install coverage tools (`gcovr` is recommended; `lcov` is supported as a fallback).  

```bash
# Install gcovr (recommended - supports XML, HTML, JSON)
sudo apt-get install gcovr -y

# OR install lcov (HTML only)
sudo apt-get install lcov -y

# Build with coverage flags (relwithdebinfo recommended)
cd ../../  # Go to project root
./build.sh --clean --coverage --type relwithdebinfo

# Verify coverage build (check for .gcno files)
ls build_x86_64/src/examples/*.gcno
```

**Build Type Recommendations**  

- `relwithdebinfo`: Recommended for development (1.5–2x slower, optimized with debug info)  
- `debug`: Most detailed coverage (3–5x slower)  

### Running Tests with Coverage

Trigger coverage analysis using the following commands  

**Manual Execution** (from `tests/cpp_example/`)  

```bash
# Quick E2E with coverage (image tests only, 2-3 minutes)
pytest -m e2e -k "test_image_inference_e2e" --coverage -v

# Full E2E tests with coverage (8-10 minutes)
pytest -m e2e --coverage -v

# Run specific models with coverage
pytest -m e2e -k "yolov7" --coverage -v
```

**Unified Test Runner** (from project root)

```bash
# Quick E2E with coverage (recommended for development)
./run_tc.sh --cpp --e2e-quick --coverage

# Full E2E with coverage
./run_tc.sh --cpp --e2e --coverage
```

### Coverage Report Output

After running tests with `--coverage`, several reports are generated in `tests/cpp_example/coverage/`  

- **Console summary:** Line and branch coverage percentages.  
- **HTML report:** Located at `html/index.html`. Shows line-by-line visualization with uncovered code in **red**.  
- **XML (Cobertura) & JSON:** Timestamped for CI/CD integration.  

**View the HTML Report**  

```bash
# Open in your browser
firefox tests/cpp_example/coverage/html/index.html
# or
xdg-open tests/cpp_example/coverage/html/index.html
```

**Report Features**  

- Overall and file-by-file coverage statistics.  
- Line-by-line visualization with **uncovered code highlighted in red**.  
- Branch coverage analysis.  

**Coverage Filtering Rules**  

- **Included:** All source files within the `src/` directory.  
- **Excluded:** System headers (`/usr/*\, third_party/*, extern/*`), and the `tests/` directory itself.  

---

## Test Coverage Summary

| **Category** | **Count** | **Status** | 
|----|----|----|
| **Total Executables** | 48 | - | 
| **CLI Tests** | 48 | All binaries validated (191 tests) | 
| **E2E Tests** | 34 | Executables with `--no-display` support (68 tests + 1 prerequisite) | 
| **Image Inference** | 34 | Validated via E2E image tests | 
| **Video Inference** | 34 | Validated via E2E video tests | 
| **Exclusions** | 14| GUI-only or multi-model variants (e.g.,`yolov7+deeplabv3`) | 

---
