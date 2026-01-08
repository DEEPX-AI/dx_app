# C++ Example Test 

This directory contains CLI tests for executables in the `bin/` directory.

## Test Structure

- `test_cli_help.py`: Tests for `--help` option of all executables
- `test_cli_basic.py`: Basic CLI argument tests
- `test_e2e.py`: End-to-end integration tests (image/video inference)

## Running Tests

```bash
# Install requirements
pip install -r requirements.txt

# Run all tests (excluding slow E2E tests)
pytest -m "not e2e"

# Run only CLI tests (fast)
pytest test_cli_help.py test_cli_basic.py

# Run only E2E tests (slow, requires models and test data)
pytest test_e2e.py -v

# Run E2E for specific executable
pytest test_e2e.py -k yolov8_async

# Run only async executables E2E tests
pytest test_e2e.py -m async_exec -v

# Run only sync executables E2E tests
pytest test_e2e.py -m sync_exec -v

# Run all tests including E2E
pytest -v

# Run with verbose output
pytest -v -s
```

## Using run_tc.sh (Recommended)

The project provides a unified test runner script at the project root:

```bash
cd ../../  # Go to project root

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

**Tip:** Use `--e2e-quick` for faster validation during development (2-3 minutes vs 8-10 minutes).

## Filtering Tests by Model

You can use pytest's `-k` option to run tests for specific models:

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

## Performance Report

After running E2E tests, a performance report is automatically generated:
- Console output: Formatted table with FPS metrics
- CSV file: `performance_report_YYYYMMDD_HHMMSS.csv` in `tests/cpp_example/`

The report includes:
- E2E FPS (overall throughput)
- Read FPS (frame reading speed)
- Preprocess FPS (preprocessing speed)
- Inference FPS (model inference speed)
- Postprocess FPS (postprocessing speed)
- Bottleneck detection (marked with asterisk *)

## Code Coverage

### Prerequisites

To generate code coverage reports, you need:
1. Build executables with coverage instrumentation
2. Install coverage tools (gcovr recommended, lcov as fallback)

```bash
# Install gcovr (recommended - supports XML, HTML, JSON)
sudo apt-get install gcovr -y

# OR install lcov (HTML only)
sudo apt-get install lcov -y

# Build with coverage flags (relwithdebinfo recommended)
cd ../../  # Go to project root
./build.sh --clean --coverage --type relwithdebinfo

# Verify coverage build
ls build_x86_64/src/examples/*.gcno  # Should see .gcno files
```

**Build Type Recommendations:**
- `relwithdebinfo`: Best balance (1.5-2x slower, optimized with debug info)
- `debug`: Most detailed coverage (3-5x slower)

### Running Tests with Coverage

```bash
cd tests/cpp_example

# Quick E2E with coverage (image tests only, 2-3 minutes)
pytest -m e2e -k "test_image_inference_e2e" --coverage -v

# Full E2E tests with coverage (8-10 minutes)
pytest -m e2e --coverage -v

# Run specific models with coverage
pytest -m e2e -k "yolov7" --coverage -v
```

**Or use the unified test runner from project root:**
```bash
# Quick E2E with coverage (recommended for development)
./run_tc.sh --cpp --e2e-quick --coverage

# Full E2E with coverage
./run_tc.sh --cpp --e2e --coverage
```

### Coverage Report Output

After running tests with `--coverage`, you'll get:
- **Console summary**: Line and branch coverage percentages
- **XML report**: `tests/cpp_example/coverage/coverage_YYYYMMDD_HHMMSS.xml` (Cobertura format)
- **HTML report**: `tests/cpp_example/coverage/html/index.html`
- **JSON report**: `tests/cpp_example/coverage/coverage_YYYYMMDD_HHMMSS.json`

Open the HTML report in a browser:
```bash
firefox tests/cpp_example/coverage/html/index.html
# or
xdg-open tests/cpp_example/coverage/html/index.html
```

The reports show:
- Overall coverage statistics
- File-by-file coverage breakdown
- Line-by-line coverage visualization
- Uncovered code highlighted in red
- Branch coverage analysis

**XML files are timestamped** for tracking coverage over time and CI/CD integration.

**Coverage Filtering:**
- Includes: `src/` directory source files
- Excludes: `/usr/*`, `third_party/*`, `extern/*`, `tests/*`

**Performance Impact:**
- Debug + coverage: 3-5x slower than normal build
- RelWithDebInfo + coverage: 1.5-2x slower (recommended for development)



## Test Categories

### CLI Tests (Fast)
- `test_cli_help.py`: --help option validation for all 48 executables
- `test_cli_basic.py`: Invalid arguments and no-arguments handling

### E2E Tests (Slow)
- `test_e2e.py`: Real inference on images and videos
  - Only tests executables with `--no-display` option (36 executables)
  - Uses real models from `assets/models/`
  - Uses test data from `sample/img/` and `assets/videos/`
  - Default loop count: 50 iterations
  - Image inference timeout: 100 seconds
  - Video inference timeout: 10 minutes

## Prerequisites

### For CLI Tests
- Executables must be built in `bin/` directory
- Run `./build.sh` from project root if executables are missing

### For E2E Tests (Additional)
- Model files in `assets/models/` (run `./setup_sample_models.sh`)
- Test images in `sample/img/`
- Test videos in `assets/videos/` (run `./setup_sample_videos.sh`)
- Shared libraries in `lib/` directory

## Test Coverage

- **Total executables**: 48
- **CLI tests**: All 48 executables (191 tests)
- **E2E tests**: 34 executables with --no-display option (68 tests + 1 prerequisite)
  - Image inference: 34 tests
  - Video inference: 34 tests
  - Excluded: yolov7_x_deeplabv3_* (requires two models)
