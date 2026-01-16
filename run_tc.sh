#!/bin/bash

#########################################################
# Test Runner Script for dx_app
# 
# This script runs pytest tests for both Python and C++ examples
# - Python example: CLI, Integration, E2E, Coverage tests
# - C++ example: CLI, E2E tests
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

# Test results
PYTHON_CLI_RESULT=0
PYTHON_INTEGRATION_RESULT=0
PYTHON_E2E_RESULT=0
PYTHON_COVERAGE_RESULT=0
CPP_CLI_RESULT=0
CPP_E2E_RESULT=0

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
    RUN_INTEGRATION=true
    RUN_E2E=true
    RUN_COVERAGE=true
    
    # Test selection rules (allow combining CLI + E2E when both flags are set)
    if [ "$CLI_ONLY" = true ] && [ "$E2E_ONLY" = false ] && [ "$E2E_QUICK" = false ]; then
        RUN_INTEGRATION=false
        RUN_E2E=false
        RUN_COVERAGE=false
    elif [ "$E2E_ONLY" = true ]; then
        RUN_INTEGRATION=false
        RUN_COVERAGE=false
        if [ "$CLI_ONLY" = false ]; then
            RUN_CLI=false
        fi
    fi
    
    if [ "$SKIP_COVERAGE" = true ]; then
        RUN_COVERAGE=false
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
    
    # 2. Integration Tests
    if [ "$RUN_INTEGRATION" = true ]; then
        print_info "Running Python Integration tests..."
        if pytest -m integration --tb=short -v; then
            PYTHON_INTEGRATION_RESULT=0
            print_success "Python Integration tests passed"
        else
            PYTHON_INTEGRATION_RESULT=$?
            print_error "Python Integration tests failed"
        fi
    fi
    
    # 3. E2E Tests
    if [ "$RUN_E2E" = true ]; then
        print_info "Running Python E2E tests..."
        
        if [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: Image tests only"
            if pytest -m e2e -k "test_image" --tb=short -v; then
                PYTHON_E2E_RESULT=0
                print_success "Python E2E image tests passed"
            else
                PYTHON_E2E_RESULT=$?
                print_error "Python E2E image tests failed"
            fi
        else
            if pytest -m e2e --tb=short -v; then
                PYTHON_E2E_RESULT=0
                print_success "Python E2E tests passed"
            else
                PYTHON_E2E_RESULT=$?
                print_error "Python E2E tests failed"
            fi
        fi
    fi
    
    # 4. Coverage Tests (all tests with coverage)
    if [ "$RUN_COVERAGE" = true ]; then
        print_info "Running Python tests with coverage..."
        if pytest --cov=../../src/python_example --cov-report=term-missing:skip-covered --cov-report=html:../../htmlcov --tb=short; then
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
    
    # Handle test selection flags
    if [ "$CLI_ONLY" = true ] && [ "$E2E_ONLY" = false ] && [ "$E2E_QUICK" = false ]; then
        # Only CLI flag specified
        RUN_E2E=false
    elif [ "$E2E_ONLY" = true ] || [ "$E2E_QUICK" = true ]; then
        # E2E flags specified
        if [ "$CLI_ONLY" = false ]; then
            # Only E2E, no CLI
            RUN_CLI=false
        fi
        # If both CLI and E2E flags are set, run both
    fi
    
    # Check for coverage build if --coverage is specified
    if [ "$CPP_COVERAGE" = true ]; then
        print_info "Code coverage mode enabled"
        
        # Check for .gcno files (indicates coverage build)
        GCNO_FILES=$(find "${SCRIPT_DIR}/build"* -name "*.gcno" 2>/dev/null | wc -l)
        
        if [ "$GCNO_FILES" -eq 0 ]; then
            print_warning "No coverage instrumentation found (.gcno files missing)"
            print_warning "Please rebuild with coverage flags:"
            print_warning "  ./build.sh --clean --coverage --type relwithdebinfo"
            print_warning ""
            print_warning "Continuing without coverage report..."
            CPP_COVERAGE=false
        else
            print_info "Found coverage instrumentation ($GCNO_FILES .gcno files)"
        fi
    fi

    # If coverage is on and both CLI/E2E are requested, run them together once
    if [ "$CPP_COVERAGE" = true ] && [ "$RUN_CLI" = true ] && [ "$RUN_E2E" = true ]; then
        print_info "Running C++ CLI+E2E in one pytest run for merged coverage..."
        PYTEST_COMBINED_CMD=(pytest -m "cli or e2e" --tb=short -v --coverage)

        if [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: Image tests only for E2E"
            PYTEST_COMBINED_CMD+=(-k "test_image_inference_e2e or cli")
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
        if [ "$CPP_COVERAGE" = true ]; then
            PYTEST_CLI_CMD+=(--coverage)
        fi
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
        PYTEST_E2E_CMD=(pytest -m e2e --tb=short -v)
        if [ "$E2E_QUICK" = true ]; then
            print_info "Quick mode: Image tests only"
            PYTEST_E2E_CMD+=(-k test_image_inference_e2e)
        fi
        if [ "$CPP_COVERAGE" = true ]; then
            PYTEST_E2E_CMD+=(--coverage)
        fi
        if [ -n "$LOOP_COUNT" ]; then
            PYTEST_E2E_CMD+=(--loop "$LOOP_COUNT")
        fi

        if "${PYTEST_E2E_CMD[@]}"; then
            CPP_E2E_RESULT=0
            if [ "$E2E_QUICK" = true ]; then
                print_success "C++ E2E image tests passed"
            else
                print_success "C++ E2E tests passed"
            fi
        else
            CPP_E2E_RESULT=$?
            if [ "$E2E_QUICK" = true ]; then
                print_error "C++ E2E image tests failed"
            else
                print_error "C++ E2E tests failed"
            fi
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
        if [ $PYTHON_CLI_RESULT -eq 0 ]; then
            print_success "  CLI Tests: PASSED"
        else
            print_error "  CLI Tests: FAILED (exit code: $PYTHON_CLI_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_CLI_RESULT))
        fi
        
        if [ $PYTHON_INTEGRATION_RESULT -eq 0 ]; then
            print_success "  Integration Tests: PASSED"
        else
            print_error "  Integration Tests: FAILED (exit code: $PYTHON_INTEGRATION_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_INTEGRATION_RESULT))
        fi
        
        if [ $PYTHON_E2E_RESULT -eq 0 ]; then
            print_success "  E2E Tests: PASSED"
        else
            print_error "  E2E Tests: FAILED (exit code: $PYTHON_E2E_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_E2E_RESULT))
        fi
        
        if [ $PYTHON_COVERAGE_RESULT -eq 0 ]; then
            print_success "  Coverage Tests: PASSED"
        else
            print_error "  Coverage Tests: FAILED (exit code: $PYTHON_COVERAGE_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + PYTHON_COVERAGE_RESULT))
        fi
        
        echo ""
    fi
    
    # Only show C++ results if not in Python only mode
    if [ "$PYTHON_ONLY" != true ]; then
        echo "C++ Example Tests:"
        if [ $CPP_CLI_RESULT -eq 0 ]; then
            print_success "  CLI Tests: PASSED"
        else
            print_error "  CLI Tests: FAILED (exit code: $CPP_CLI_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_CLI_RESULT))
        fi
        
        if [ $CPP_E2E_RESULT -eq 0 ]; then
            print_success "  E2E Tests: PASSED"
        else
            print_error "  E2E Tests: FAILED (exit code: $CPP_E2E_RESULT)"
            TOTAL_FAILURES=$((TOTAL_FAILURES + CPP_E2E_RESULT))
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
SKIP_COVERAGE=false
QUICK_MODE=false
CPP_COVERAGE=false
CLI_ONLY=false
E2E_ONLY=false
E2E_QUICK=false
LOOP_COUNT=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --python          Run only Python example tests"
    echo "  --cpp             Run only C++ example tests"
    echo "  --cli             Run only CLI tests (can combine with --python or --cpp)"
    echo "  --e2e             Run only E2E tests (can combine with --python or --cpp)"
    echo "  --e2e-quick       Run only E2E image tests (faster, skips video tests)"
    echo "  --skip-coverage   Skip coverage tests for Python"
    echo "  --coverage        Generate C++ code coverage report (requires --cpp or full run)"
    echo "  --loop <N>        Override loop count for C++ E2E image tests (default: 50)"
    echo "  --quick           Run only CLI tests (no E2E) - same as --cli"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run all tests"
    echo "  $0 --python              # Run only Python tests"
    echo "  $0 --cpp                 # Run only C++ tests"
    echo "  $0 --cpp --cli           # Run only C++ CLI tests"
    echo "  $0 --cpp --e2e           # Run only C++ E2E tests"
    echo "  $0 --cpp --e2e-quick     # Run only C++ E2E image tests (fast)"
    echo "  $0 --cpp --e2e --coverage # Run C++ E2E tests with coverage"
    echo "  $0 --python --e2e        # Run only Python E2E tests"
    echo "  $0 --skip-coverage       # Run all tests but skip Python coverage"
    echo "  $0 --quick               # Quick mode: CLI tests only"
    echo ""
    echo "Note: C++ coverage requires executables built with --coverage flag:"
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
        --skip-coverage)
            SKIP_COVERAGE=true
            shift
            ;;
        --coverage)
            CPP_COVERAGE=true
            shift
            ;;
        --cli)
            CLI_ONLY=true
            shift
            ;;
        --e2e)
            E2E_ONLY=true
            shift
            ;;
        --e2e-quick)
            E2E_ONLY=true
            E2E_QUICK=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            CLI_ONLY=true  # --quick is same as --cli
            shift
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

# Validate options - allow --cli with --e2e-quick for combined testing
if [ "$CLI_ONLY" = true ] && [ "$E2E_ONLY" = true ] && [ "$E2E_QUICK" = false ]; then
    print_error "Cannot specify both --cli and --e2e"
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
