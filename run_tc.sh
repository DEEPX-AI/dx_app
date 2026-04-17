#!/bin/bash

#########################################################
# Test Runner Script for dx_app
#
# Runs pytest-based tests for Python and C++ examples.
# Options: --cli, --e2e, --e2e-quick, --e2e-short, --vis,
#          --signal, --coverage, --loop <N>,
#          --camera, --camera-index <N>, --rtsp, --rtsp-url <URL>,
#          --stream-duration <N>
# Scope:   --python, --cpp (default: both)
#########################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/tests"

# Test results (-1 = not run, 0 = passed, >0 = failed)
PYTHON_CLI_RESULT=-1
PYTHON_E2E_RESULT=-1
PYTHON_VIS_RESULT=-1
PYTHON_SIGNAL_RESULT=-1
PYTHON_COVERAGE_RESULT=-1
PYTHON_CAMERA_RESULT=-1
PYTHON_RTSP_RESULT=-1
CPP_CLI_RESULT=-1
CPP_E2E_RESULT=-1
CPP_VIS_RESULT=-1
CPP_SIGNAL_RESULT=-1
CPP_CAMERA_RESULT=-1
CPP_RTSP_RESULT=-1

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to run Python example tests
run_python_tests() {
    print_header "Running Python Example Tests"

    cd "${TEST_DIR}/python_example"

    # Check if dependencies are installed
    if ! python -c "import pytest" &> /dev/null; then
        print_warning "pytest not found. Installing requirements..."
        pip install -r requirements.txt
    fi

    # Determine which tests to run
    RUN_CLI=true
    RUN_E2E=true
    RUN_VIS=false
    RUN_COVERAGE=false

    if [ "$COVERAGE" = true ]; then
        RUN_COVERAGE=true
    fi

    # Camera/RTSP mode: skip normal CLI/E2E, only run camera/rtsp sections
    if [ "$CAMERA_MODE" = true ] || [ "$RTSP_MODE" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
        RUN_COVERAGE=false
    fi

    # Test selection rules (allow combining CLI + E2E when both flags are set)
    if [ "$VIS_ONLY" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
        RUN_VIS=true
        RUN_COVERAGE=false
    elif [ "$SIGNAL_ONLY" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
        RUN_COVERAGE=false
    elif [ "$CLI_ONLY" = true ] && [ "$E2E_ONLY" = false ] && [ "$E2E_QUICK" = false ] && [ "$E2E_SHORT" = false ]; then
        RUN_E2E=false
        RUN_COVERAGE=false
    elif [ "$E2E_ONLY" = true ]; then
        RUN_COVERAGE=false
        if [ "$CLI_ONLY" = false ]; then
            RUN_CLI=false
        fi
    fi

    # Coverage already includes CLI + E2E, skip separate runs to avoid duplication
    if [ "$RUN_COVERAGE" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
    fi

    # 1. CLI Tests
    if [ "$RUN_CLI" = true ]; then
        print_info "Running Python CLI tests..."
        if pytest -m cli --tb=short -v; then
            PYTHON_CLI_RESULT=0
            print_success "Python CLI tests passed"
        else
            PYTHON_CLI_RESULT=$?
            print_error "Python CLI tests failed"
        fi
    fi

    # 2. E2E Tests
    if [ "$RUN_E2E" = true ]; then
        print_info "Running Python E2E tests..."

        if [ "$E2E_SHORT" = true ]; then
            print_info "Short mode: representative models, stream tests"
            PYTEST_PY_E2E_CMD=(pytest -m "e2e_stream and e2e_short" --tb=short -v)
        elif [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: Image tests only"
            PYTEST_PY_E2E_CMD=(pytest -m e2e_image --tb=short -v)
        else
            print_info "Full mode: Stream tests for all models"
            PYTEST_PY_E2E_CMD=(pytest -m e2e_stream --tb=short -v)
        fi

        if [ -n "$LOOP_COUNT" ]; then
            PYTEST_PY_E2E_CMD+=(--loop "$LOOP_COUNT")
        fi

        PYTHON_E2E_RESULT=0
        "${PYTEST_PY_E2E_CMD[@]}" || PYTHON_E2E_RESULT=$?
        # Exit code 5 = no tests collected (no models installed) — treat as pass
        if [ "$PYTHON_E2E_RESULT" -eq 0 ] || [ "$PYTHON_E2E_RESULT" -eq 5 ]; then
            PYTHON_E2E_RESULT=0
            if [ "$E2E_SHORT" = true ]; then
                print_success "Python E2E short tests passed"
            elif [ "$E2E_QUICK" = true ]; then
                print_success "Python E2E image tests passed"
            else
                print_success "Python E2E stream tests passed"
            fi
        else
            if [ "$E2E_SHORT" = true ]; then
                print_error "Python E2E short tests failed"
            elif [ "$E2E_QUICK" = true ]; then
                print_error "Python E2E image tests failed"
            else
                print_error "Python E2E stream tests failed"
            fi
        fi
    fi

    # 3. Visualization Tests
    if [ "$RUN_VIS" = true ]; then
        print_info "Running Python Visualization tests..."
        if pytest -m visualization --tb=short -v; then
            PYTHON_VIS_RESULT=0
            print_success "Python Visualization tests passed"
        else
            PYTHON_VIS_RESULT=$?
            print_error "Python Visualization tests failed"
        fi
    fi

    # 4. Signal Handling Tests
    if [ "$SIGNAL_ONLY" = true ]; then
        print_info "Running Python Signal Handling tests..."
        PYTHON_SIGNAL_RESULT=0
        pytest -m signal_handling --tb=short -v || PYTHON_SIGNAL_RESULT=$?
        if [ "$PYTHON_SIGNAL_RESULT" -eq 0 ]; then
            print_success "Python Signal Handling tests passed"
        else
            print_error "Python Signal Handling tests failed"
        fi
    fi

    # 5. Camera Tests
    if [ "$CAMERA_MODE" = true ]; then
        print_info "Running Python Camera tests (camera-index=${CAMERA_INDEX})..."
        PYTEST_CAM_CMD=(pytest test_e2e_camera_rtsp.py -m e2e_camera --camera-index "$CAMERA_INDEX" --tb=short -v)
        if [ "$E2E_SHORT" = true ]; then
            PYTEST_CAM_CMD=(pytest test_e2e_camera_rtsp.py -m "e2e_camera and e2e_short" --camera-index "$CAMERA_INDEX" --tb=short -v)
        fi
        if [ -n "$STREAM_DURATION" ]; then
            PYTEST_CAM_CMD+=(--stream-duration "$STREAM_DURATION")
        fi
        PYTHON_CAMERA_RESULT=0
        "${PYTEST_CAM_CMD[@]}" || PYTHON_CAMERA_RESULT=$?
        if [ "$PYTHON_CAMERA_RESULT" -eq 0 ] || [ "$PYTHON_CAMERA_RESULT" -eq 5 ]; then
            PYTHON_CAMERA_RESULT=0
            print_success "Python Camera tests passed"
        else
            print_error "Python Camera tests failed"
        fi
    fi

    # 6. RTSP Tests
    if [ "$RTSP_MODE" = true ]; then
        print_info "Running Python RTSP tests (url=${RTSP_URL})..."
        PYTEST_RTSP_CMD=(pytest test_e2e_camera_rtsp.py -m e2e_rtsp --rtsp-url "$RTSP_URL" --tb=short -v)
        if [ "$E2E_SHORT" = true ]; then
            PYTEST_RTSP_CMD=(pytest test_e2e_camera_rtsp.py -m "e2e_rtsp and e2e_short" --rtsp-url "$RTSP_URL" --tb=short -v)
        fi
        if [ -n "$STREAM_DURATION" ]; then
            PYTEST_RTSP_CMD+=(--stream-duration "$STREAM_DURATION")
        fi
        PYTHON_RTSP_RESULT=0
        "${PYTEST_RTSP_CMD[@]}" || PYTHON_RTSP_RESULT=$?
        if [ "$PYTHON_RTSP_RESULT" -eq 0 ] || [ "$PYTHON_RTSP_RESULT" -eq 5 ]; then
            PYTHON_RTSP_RESULT=0
            print_success "Python RTSP tests passed"
        else
            print_error "Python RTSP tests failed"
        fi
    fi

    # 5. Coverage Tests (all tests with coverage)
    if [ "$RUN_COVERAGE" = true ]; then
        print_info "Running Python tests with coverage..."
        if [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: all tests except visualization"
            PYTEST_COV_CMD=(pytest -m "not visualization and not multi_loop and not signal_handling")
        else
            PYTEST_COV_CMD=(pytest -m "not e2e_stream and not visualization and not multi_loop and not signal_handling")
        fi
        PYTEST_COV_CMD+=(--cov=../../src/python_example --cov-config="${SCRIPT_DIR}/.coveragerc" --cov-report=term-missing:skip-covered --cov-report=html:../../htmlcov --tb=short)
        if "${PYTEST_COV_CMD[@]}"; then
            PYTHON_COVERAGE_RESULT=0
            print_success "Python coverage tests passed"
            print_info "Coverage report generated at: ${SCRIPT_DIR}/htmlcov/index.html"
        else
            PYTHON_COVERAGE_RESULT=$?
            print_error "Python coverage tests failed"
        fi
    fi

    cd "${SCRIPT_DIR}"
}

# Function to run C++ example tests
run_cpp_tests() {
    print_header "Running C++ Example Tests"

    cd "${TEST_DIR}/cpp_example"

    # Check if dependencies are installed
    if ! python -c "import pytest" &> /dev/null; then
        print_warning "pytest not found. Installing requirements..."
        pip install -r requirements.txt
    fi

    # Determine which tests to run
    RUN_CLI=true
    RUN_E2E=true
    RUN_VIS=false

    # Camera/RTSP mode: skip normal CLI/E2E, only run camera/rtsp sections
    if [ "$CAMERA_MODE" = true ] || [ "$RTSP_MODE" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
    fi

    # Handle test selection flags
    if [ "$VIS_ONLY" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
        RUN_VIS=true
    elif [ "$SIGNAL_ONLY" = true ]; then
        RUN_CLI=false
        RUN_E2E=false
    elif [ "$CLI_ONLY" = true ] && [ "$E2E_ONLY" = false ] && [ "$E2E_QUICK" = false ] && [ "$E2E_SHORT" = false ]; then
        # Only CLI flag specified
        RUN_E2E=false
    elif [ "$E2E_ONLY" = true ] || [ "$E2E_QUICK" = true ] || [ "$E2E_SHORT" = true ]; then
        # E2E flags specified
        if [ "$CLI_ONLY" = false ]; then
            # Only E2E, no CLI
            RUN_CLI=false
        fi
        # If both CLI and E2E flags are set, run both
    fi

    # Check for coverage build if --coverage is specified
    if [ "$COVERAGE" = true ]; then
        print_info "Code coverage mode enabled"

        # Check for .gcno files (indicates coverage build)
        GCNO_FILES=$(find "${SCRIPT_DIR}/build"* -name "*.gcno" 2>/dev/null | wc -l)

        if [ "$GCNO_FILES" -eq 0 ]; then
            print_warning "No coverage instrumentation found (.gcno files missing)"
            print_warning "Please rebuild with coverage flags:"
            print_warning "  ./build.sh --clean --coverage --type relwithdebinfo"
            print_warning ""
            print_warning "Continuing without coverage report..."
            COVERAGE=false
        else
            print_info "Found coverage instrumentation ($GCNO_FILES .gcno files)"
        fi
    fi

    # If coverage is on and both CLI/E2E are requested, run them together once
    if [ "$COVERAGE" = true ] && [ "$RUN_CLI" = true ] && [ "$RUN_E2E" = true ]; then
        print_info "Running C++ tests in one pytest run for merged coverage..."
        if [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: all tests except visualization"
            PYTEST_COMBINED_CMD=(pytest -m "not visualization and not multi_loop and not signal_handling" --tb=short -v --coverage)
        else
            PYTEST_COMBINED_CMD=(pytest -m "not e2e_stream and not visualization and not multi_loop and not signal_handling" --tb=short -v --coverage)
        fi

        if [ -n "$LOOP_COUNT" ]; then
            PYTEST_COMBINED_CMD+=(--loop "$LOOP_COUNT")
        fi

        if "${PYTEST_COMBINED_CMD[@]}"; then
            CPP_CLI_RESULT=0
            CPP_E2E_RESULT=0
            print_success "C++ CLI+E2E tests passed"
        else
            COMBINED_STATUS=$?
            CPP_CLI_RESULT=$COMBINED_STATUS
            CPP_E2E_RESULT=$COMBINED_STATUS
            print_error "C++ CLI+E2E tests failed"
        fi

        # Skip separate runs; results already recorded
        RUN_CLI=false
        RUN_E2E=false
    fi

    # 1. CLI Tests
    if [ "$RUN_CLI" = true ]; then
        print_info "Running C++ CLI tests..."
        PYTEST_CLI_CMD=(pytest -m cli --tb=short -v)
        if [ -n "$LOOP_COUNT" ]; then
            PYTEST_CLI_CMD+=(--loop "$LOOP_COUNT")
        fi

        if "${PYTEST_CLI_CMD[@]}"; then
            CPP_CLI_RESULT=0
            print_success "C++ CLI tests passed"
        else
            CPP_CLI_RESULT=$?
            print_error "C++ CLI tests failed"
        fi
    fi

    # 2. E2E Tests
    if [ "$RUN_E2E" = true ]; then
        print_info "Running C++ E2E tests..."

        # Build pytest command
        if [ "$E2E_SHORT" = true ]; then
            print_info "Short mode: representative models, stream tests"
            PYTEST_E2E_CMD=(pytest -m "e2e_stream and e2e_short" --tb=short -v)
        elif [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: Image tests only"
            PYTEST_E2E_CMD=(pytest -m e2e_image --tb=short -v)
        else
            print_info "Full mode: Stream tests for all models"
            PYTEST_E2E_CMD=(pytest -m e2e_stream --tb=short -v)
        fi
        if [ -n "$LOOP_COUNT" ]; then
            PYTEST_E2E_CMD+=(--loop "$LOOP_COUNT")
        fi

        CPP_E2E_RESULT=0
        "${PYTEST_E2E_CMD[@]}" || CPP_E2E_RESULT=$?
        # Exit code 5 = no tests collected (no models installed) — treat as pass
        if [ "$CPP_E2E_RESULT" -eq 0 ] || [ "$CPP_E2E_RESULT" -eq 5 ]; then
            CPP_E2E_RESULT=0
            if [ "$E2E_SHORT" = true ]; then
                print_success "C++ E2E short tests passed"
            elif [ "$E2E_QUICK" = true ]; then
                print_success "C++ E2E image tests passed"
            else
                print_success "C++ E2E stream tests passed"
            fi
        else
            if [ "$E2E_SHORT" = true ]; then
                print_error "C++ E2E short tests failed"
            elif [ "$E2E_QUICK" = true ]; then
                print_error "C++ E2E image tests failed"
            else
                print_error "C++ E2E stream tests failed"
            fi
        fi
    fi

    # 3. Visualization Tests
    if [ "$RUN_VIS" = true ]; then
        print_info "Running C++ Visualization tests..."
        PYTEST_VIS_CMD=(pytest -m visualization --tb=short -v)

        if "${PYTEST_VIS_CMD[@]}"; then
            CPP_VIS_RESULT=0
            print_success "C++ Visualization tests passed"
        else
            CPP_VIS_RESULT=$?
            print_error "C++ Visualization tests failed"
        fi
    fi

    # 4. Signal Handling Tests
    if [ "$SIGNAL_ONLY" = true ]; then
        print_info "Running C++ Signal Handling tests..."
        CPP_SIGNAL_RESULT=0
        pytest -m signal_handling --tb=short -v || CPP_SIGNAL_RESULT=$?
        if [ "$CPP_SIGNAL_RESULT" -eq 0 ]; then
            print_success "C++ Signal Handling tests passed"
        else
            print_error "C++ Signal Handling tests failed"
        fi
    fi

    # 5. Camera Tests
    if [ "$CAMERA_MODE" = true ]; then
        print_info "Running C++ Camera tests (camera-index=${CAMERA_INDEX})..."
        PYTEST_CAM_CMD=(pytest test_e2e_camera_rtsp.py -m e2e_camera --camera-index "$CAMERA_INDEX" --tb=short -v)
        if [ "$E2E_SHORT" = true ]; then
            PYTEST_CAM_CMD=(pytest test_e2e_camera_rtsp.py -m "e2e_camera and e2e_short" --camera-index "$CAMERA_INDEX" --tb=short -v)
        fi
        if [ -n "$STREAM_DURATION" ]; then
            PYTEST_CAM_CMD+=(--stream-duration "$STREAM_DURATION")
        fi
        CPP_CAMERA_RESULT=0
        "${PYTEST_CAM_CMD[@]}" || CPP_CAMERA_RESULT=$?
        if [ "$CPP_CAMERA_RESULT" -eq 0 ] || [ "$CPP_CAMERA_RESULT" -eq 5 ]; then
            CPP_CAMERA_RESULT=0
            print_success "C++ Camera tests passed"
        else
            print_error "C++ Camera tests failed"
        fi
    fi

    # 6. RTSP Tests
    if [ "$RTSP_MODE" = true ]; then
        print_info "Running C++ RTSP tests (url=${RTSP_URL})..."
        PYTEST_RTSP_CMD=(pytest test_e2e_camera_rtsp.py -m e2e_rtsp --rtsp-url "$RTSP_URL" --tb=short -v)
        if [ "$E2E_SHORT" = true ]; then
            PYTEST_RTSP_CMD=(pytest test_e2e_camera_rtsp.py -m "e2e_rtsp and e2e_short" --rtsp-url "$RTSP_URL" --tb=short -v)
        fi
        if [ -n "$STREAM_DURATION" ]; then
            PYTEST_RTSP_CMD+=(--stream-duration "$STREAM_DURATION")
        fi
        CPP_RTSP_RESULT=0
        "${PYTEST_RTSP_CMD[@]}" || CPP_RTSP_RESULT=$?
        if [ "$CPP_RTSP_RESULT" -eq 0 ] || [ "$CPP_RTSP_RESULT" -eq 5 ]; then
            CPP_RTSP_RESULT=0
            print_success "C++ RTSP tests passed"
        else
            print_error "C++ RTSP tests failed"
        fi
    fi

    cd "${SCRIPT_DIR}"
}

# Function to print summary
print_summary() {
    print_header "Test Summary"

    TOTAL_FAILURES=0

    # Only show Python results if not in C++ only mode
    if [ "$CPP_ONLY" != true ]; then
        echo "Python Example Tests:"
        if [ $PYTHON_CLI_RESULT -eq -1 ]; then
            echo -e "  CLI Tests: SKIPPED"
        elif [ $PYTHON_CLI_RESULT -eq 0 ]; then
            print_success "  CLI Tests: PASSED"
        else
            print_error "  CLI Tests: FAILED (exit code: $PYTHON_CLI_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_CLI_RESULT))
        fi

        if [ $PYTHON_E2E_RESULT -eq -1 ]; then
            echo -e "  E2E Tests: SKIPPED"
        elif [ $PYTHON_E2E_RESULT -eq 0 ]; then
            print_success "  E2E Tests: PASSED"
        else
            print_error "  E2E Tests: FAILED (exit code: $PYTHON_E2E_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_E2E_RESULT))
        fi

        if [ $PYTHON_VIS_RESULT -eq -1 ]; then
            echo -e "  Visualization Tests: SKIPPED"
        elif [ $PYTHON_VIS_RESULT -eq 0 ]; then
            print_success "  Visualization Tests: PASSED"
        else
            print_error "  Visualization Tests: FAILED (exit code: $PYTHON_VIS_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_VIS_RESULT))
        fi

        if [ $PYTHON_SIGNAL_RESULT -eq -1 ]; then
            echo -e "  Signal Handling Tests: SKIPPED"
        elif [ $PYTHON_SIGNAL_RESULT -eq 0 ]; then
            print_success "  Signal Handling Tests: PASSED"
        else
            print_error "  Signal Handling Tests: FAILED (exit code: $PYTHON_SIGNAL_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_SIGNAL_RESULT))
        fi

        if [ $PYTHON_COVERAGE_RESULT -eq -1 ]; then
            echo -e "  Coverage Tests: SKIPPED"
        elif [ $PYTHON_COVERAGE_RESULT -eq 0 ]; then
            print_success "  Coverage Tests: PASSED"
        else
            print_error "  Coverage Tests: FAILED (exit code: $PYTHON_COVERAGE_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_COVERAGE_RESULT))
        fi

        if [ $PYTHON_CAMERA_RESULT -eq -1 ]; then
            echo -e "  Camera Tests: SKIPPED"
        elif [ $PYTHON_CAMERA_RESULT -eq 0 ]; then
            print_success "  Camera Tests: PASSED"
        else
            print_error "  Camera Tests: FAILED (exit code: $PYTHON_CAMERA_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_CAMERA_RESULT))
        fi

        if [ $PYTHON_RTSP_RESULT -eq -1 ]; then
            echo -e "  RTSP Tests: SKIPPED"
        elif [ $PYTHON_RTSP_RESULT -eq 0 ]; then
            print_success "  RTSP Tests: PASSED"
        else
            print_error "  RTSP Tests: FAILED (exit code: $PYTHON_RTSP_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_RTSP_RESULT))
        fi

        echo ""
    fi

    # Only show C++ results if not in Python only mode
    if [ "$PYTHON_ONLY" != true ]; then
        echo "C++ Example Tests:"
        if [ $CPP_CLI_RESULT -eq -1 ]; then
            echo -e "  CLI Tests: SKIPPED"
        elif [ $CPP_CLI_RESULT -eq 0 ]; then
            print_success "  CLI Tests: PASSED"
        else
            print_error "  CLI Tests: FAILED (exit code: $CPP_CLI_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_CLI_RESULT))
        fi

        if [ $CPP_E2E_RESULT -eq -1 ]; then
            echo -e "  E2E Tests: SKIPPED"
        elif [ $CPP_E2E_RESULT -eq 0 ]; then
            print_success "  E2E Tests: PASSED"
        else
            print_error "  E2E Tests: FAILED (exit code: $CPP_E2E_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_E2E_RESULT))
        fi

        if [ $CPP_VIS_RESULT -eq -1 ]; then
            echo -e "  Visualization Tests: SKIPPED"
        elif [ $CPP_VIS_RESULT -eq 0 ]; then
            print_success "  Visualization Tests: PASSED"
        else
            print_error "  Visualization Tests: FAILED (exit code: $CPP_VIS_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_VIS_RESULT))
        fi

        if [ $CPP_SIGNAL_RESULT -eq -1 ]; then
            echo -e "  Signal Handling Tests: SKIPPED"
        elif [ $CPP_SIGNAL_RESULT -eq 0 ]; then
            print_success "  Signal Handling Tests: PASSED"
        else
            print_error "  Signal Handling Tests: FAILED (exit code: $CPP_SIGNAL_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_SIGNAL_RESULT))
        fi

        if [ $CPP_CAMERA_RESULT -eq -1 ]; then
            echo -e "  Camera Tests: SKIPPED"
        elif [ $CPP_CAMERA_RESULT -eq 0 ]; then
            print_success "  Camera Tests: PASSED"
        else
            print_error "  Camera Tests: FAILED (exit code: $CPP_CAMERA_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_CAMERA_RESULT))
        fi

        if [ $CPP_RTSP_RESULT -eq -1 ]; then
            echo -e "  RTSP Tests: SKIPPED"
        elif [ $CPP_RTSP_RESULT -eq 0 ]; then
            print_success "  RTSP Tests: PASSED"
        else
            print_error "  RTSP Tests: FAILED (exit code: $CPP_RTSP_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_RTSP_RESULT))
        fi

        echo ""
    fi

    if [ $TOTAL_FAILURES -eq 0 ]; then
        print_success "All tests passed! ✓"
        return 0
    else
        print_error "Some tests failed. Total failures: $TOTAL_FAILURES"
        return 1
    fi
}

# Parse command line arguments
PYTHON_ONLY=false
CPP_ONLY=false
QUICK_MODE=false
COVERAGE=false
CLI_ONLY=false
E2E_ONLY=false
E2E_QUICK=false
E2E_SHORT=false
LOOP_COUNT=""
VIS_ONLY=false
SIGNAL_ONLY=false
CAMERA_MODE=false
RTSP_MODE=false
CAMERA_INDEX=""
RTSP_URL=""
STREAM_DURATION=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --python          Run only Python example tests"
    echo "  --cpp             Run only C++ example tests"
    echo "  --cli             Run CLI tests (can combine with --e2e-quick or --e2e-short)"
    echo "  --e2e             Run E2E stream tests for all models"
    echo "  --e2e-quick       Run E2E image tests only (faster, skips stream tests)"
    echo "  --e2e-short       Run E2E stream tests for representative models"
    echo "  --vis             Run visualization tests (can combine with --python or --cpp)"
    echo "  --signal          Run signal handling tests (SIGINT graceful shutdown)"
    echo "  --coverage        Run all tests with coverage for SonarQube (standalone option)"
    echo "  --camera          Run camera input inference tests (requires --camera-index)"
    echo "  --camera-index <N>  Camera device index (e.g., 0)"
    echo "  --rtsp            Run RTSP input inference tests (requires --rtsp-url)"
    echo "  --rtsp-url <URL>  RTSP stream URL (e.g., rtsp://192.168.30.100:8554/stream1)"
    echo "  --stream-duration <N>  Seconds to run each camera/RTSP test (default: 10)"
    echo "  --loop <N>        Override loop count for E2E image tests (default: C++ 50, Python 1)"
    echo "  --quick           Same as --cli"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Default behavior (no flags): CLI + E2E stream tests for all models"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run CLI + E2E stream tests (all models)"
    echo "  $0 --python              # Run Python CLI + E2E stream tests"
    echo "  $0 --cpp                 # Run C++ CLI + E2E stream tests"
    echo "  $0 --cli                 # Run CLI tests only (no E2E)"
    echo "  $0 --cpp --cli           # Run only C++ CLI tests"
    echo "  $0 --e2e                 # Run E2E stream tests only (no CLI)"
    echo "  $0 --e2e-quick           # Run E2E image tests only (no CLI)"
    echo "  $0 --cli --e2e-quick     # Run CLI + E2E image tests"
    echo "  $0 --cpp --e2e-quick     # Run C++ E2E image tests only"
    echo "  $0 --e2e-short           # Run E2E stream tests for representative models"
    echo "  $0 --vis                 # Run Python + C++ visualization tests"
    echo "  $0 --cpp --vis           # Run C++ visualization tests"
    echo "  $0 --signal              # Run signal handling tests"
    echo "  $0 --camera --camera-index 0                  # Camera tests (all models)"
    echo "  $0 --cpp --camera --camera-index 0             # C++ camera tests only"
    echo "  $0 --rtsp --rtsp-url rtsp://host:8554/stream   # RTSP tests (all models)"
    echo "  $0 --camera --rtsp --camera-index 0 --rtsp-url rtsp://host:8554/stream  # Both"
    echo "  $0 --camera --camera-index 0 --e2e-short       # Camera tests for representative models"
    echo "  $0 --coverage            # Run all tests with coverage (Python + C++)"
    echo "  $0 --cpp --coverage      # Run C++ tests with coverage"
    echo "  $0 --python --coverage   # Run Python tests with coverage"
    echo ""
    echo "Note: --coverage cannot be combined with --cli, --e2e, --vis, --signal, --loop, etc."
    echo "      C++ coverage requires executables built with:"
    echo "      ./build.sh --clean --coverage --type relwithdebinfo"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_ONLY=true
            shift
            ;;
        --cpp)
            CPP_ONLY=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --cli)
            CLI_ONLY=true
            shift
            ;;
        --e2e)
            E2E_ONLY=true
            E2E_QUICK=false
            shift
            ;;
        --e2e-quick)
            E2E_ONLY=true
            E2E_QUICK=true
            shift
            ;;
        --e2e-short)
            E2E_ONLY=true
            E2E_SHORT=true
            E2E_QUICK=false
            shift
            ;;
        --vis|--visualization)
            VIS_ONLY=true
            shift
            ;;
        --signal|--signal-handling)
            SIGNAL_ONLY=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            CLI_ONLY=true  # --quick is same as --cli
            shift
            ;;
        --camera)
            CAMERA_MODE=true
            shift
            ;;
        --camera-index)
            CAMERA_INDEX="$2"
            shift 2
            ;;
        --rtsp)
            RTSP_MODE=true
            shift
            ;;
        --rtsp-url)
            RTSP_URL="$2"
            shift 2
            ;;
        --stream-duration)
            STREAM_DURATION="$2"
            shift 2
            ;;
        --loop)
            LOOP_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
print_header "dx_app Test Suite Runner"
print_info "Test directory: ${TEST_DIR}"

# Validate options - --coverage is a solo option (only --cpp/--python allowed)
if [ "$COVERAGE" = true ]; then
    if [ "$CLI_ONLY" = true ] || [ "$E2E_ONLY" = true ] || [ "$E2E_QUICK" = true ] || \
       [ "$E2E_SHORT" = true ] || [ "$VIS_ONLY" = true ] || [ "$SIGNAL_ONLY" = true ] || \
       [ "$QUICK_MODE" = true ] || [ -n "$LOOP_COUNT" ]; then
        print_error "--coverage is a standalone option (only --cpp/--python can be combined)"
        usage
        exit 1
    fi
fi

# Validate options - allow --cli with --e2e-quick for combined testing
if [ "$CLI_ONLY" = true ] && [ "$E2E_ONLY" = true ] && [ "$E2E_QUICK" = false ] && [ "$E2E_SHORT" = false ]; then
    print_error "Cannot specify both --cli and --e2e"
    usage
    exit 1
fi

# Validate camera/rtsp options
if [ "$CAMERA_MODE" = true ] && [ -z "$CAMERA_INDEX" ]; then
    print_error "--camera requires --camera-index <N>"
    usage
    exit 1
fi
if [ "$RTSP_MODE" = true ] && [ -z "$RTSP_URL" ]; then
    print_error "--rtsp requires --rtsp-url <URL>"
    usage
    exit 1
fi

# Run tests based on options
if [ "$PYTHON_ONLY" = true ]; then
    run_python_tests
elif [ "$CPP_ONLY" = true ]; then
    run_cpp_tests
else
    run_python_tests
    run_cpp_tests
fi

# Print summary and exit with appropriate code
print_summary
exit $?
