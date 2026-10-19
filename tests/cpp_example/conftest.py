"""pytest configuration for the C++ example suite (the ``bin/`` executables).

Scope: the CLI/help contract tests and E2E inference tests in this directory,
plus gcovr/lcov coverage reporting behind ``--coverage``. Option and fixture
wiring shared with the Python suite lives in
``tests/test_helpers/pytest_support.py`` -- see ``tests/README.md``.
"""
import os
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from performance_collector import get_collector

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
# Make tests/test_helpers importable here and in every test module of this suite.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_helpers.pytest_support import (  # noqa: E402
    add_loop_option,
    add_stream_options,
    configure_stdio_errors,
)
# Re-exported fixtures -- importing them here is what makes pytest discover them.
from test_helpers.pytest_support import (  # noqa: E402,F401
    camera_index,
    rtsp_url,
    stream_duration,
    wait_for_temperature,
)

from test_helpers.proc import apply_headless_display  # noqa: E402

configure_stdio_errors()

# No display: OpenCV's Qt backend aborts (rc 134) on a window it cannot show.
# offscreen lets every example run the same code path as with a display.
apply_headless_display(os.environ)

logger = logging.getLogger(__name__)

#: Default inference iterations for C++ image E2E (the Python suite uses 1).
#: Stream E2E does not use this; it passes ``-l 1``.
DEFAULT_LOOP_COUNT = 20


def resolve_bin_dir() -> Path:
    """Return the bin directory, handling Windows Release layout."""
    if os.name == "nt":
        # MSVC builds place binaries under bin/Release (or Debug), but
        # build.bat's install step copies them straight into bin/. Returning
        # bin/Release unconditionally made EXECUTABLE_PARAMS empty whenever the
        # exes sat in bin/ -> pytest skipped a [NOTSET] placeholder and exited 0,
        # which the req_test runner then reported as a PASS. Probe for a real
        # .exe and fall back to bin/.
        for candidate in (PROJECT_ROOT / "bin" / "Release",
                          PROJECT_ROOT / "bin" / "Debug"):
            if next(candidate.glob("*.exe"), None) is not None:
                return candidate
    return PROJECT_ROOT / "bin"

def is_executable(path: Path) -> bool:
    """Cross-platform executable check."""
    if not path.is_file():
        return False
    if os.name == "nt":
        return path.suffix.lower() == ".exe"
    return os.access(path, os.X_OK)


# Path to bin directory
BIN_DIR = resolve_bin_dir()

#: bin/ executables that are plain-assert unit-test binaries, not example apps:
#: they run their checks whatever the arguments, so the CLI-argument tests
#: (--help, invalid option, no arguments) do not apply. test_graph_engine.py
#: and test_common_unit.py run them.
UNIT_TEST_BINARIES = frozenset({"graph_engine_test", "common_unit_test"})
BUILD_DIR = PROJECT_ROOT / "build"


def pytest_addoption(parser):
    """Register this suite's own options plus those shared with the Python suite."""
    parser.addoption(
        "--coverage",
        action="store_true",
        default=False,
        help="Generate code coverage report after tests",
    )
    add_loop_option(parser, default=str(DEFAULT_LOOP_COUNT))
    add_stream_options(parser)


@pytest.fixture
def bin_dir():
    """Fixture providing path to bin directory"""
    return BIN_DIR


@pytest.fixture(scope="session")
def available_executables():
    """Fixture providing list of available executables in bin directory"""
    if not BIN_DIR.exists():
        pytest.skip(f"Bin directory not found: {BIN_DIR}")
    
    executables = [f.name for f in BIN_DIR.iterdir() if f.is_file() and f.stat().st_mode & 0o111]
    if not executables:
        pytest.skip(f"No executables found in: {BIN_DIR}")
    
    return sorted(executables)


@pytest.fixture(scope="session")
def loop_count(request) -> int:
    """Loop count for E2E image tests (overridable via --loop)."""
    try:
        return int(request.config.getoption("--loop"))
    except (TypeError, ValueError):
        return DEFAULT_LOOP_COUNT


def pytest_sessionfinish(session, exitstatus):
    """Hook called after whole test run finishes"""
    # Generate performance report
    collector = get_collector()
    report = collector.generate_report()
    
    if report:
        print("\n")
        print(report)
        
        # Generate CSV file
        csv_path = collector.generate_csv()
        print(f"\nPerformance data saved to: {csv_path}")
        print(f"{csv_path.absolute()}\n")
    
    # Generate code coverage report if --coverage flag is set
    if session.config.getoption("--coverage"):
        generate_coverage_report()


def generate_coverage_report():
    """Generate code coverage report using lcov and gcovr"""
    print("\n" + "=" * 80)
    print("Generating Code Coverage Report")
    print("=" * 80 + "\n")
    
    # Check if lcov or gcovr is installed (prefer gcovr for XML support)
    has_gcovr = False
    has_lcov = False
    
    try:
        subprocess.run(["gcovr", "--version"], capture_output=True, check=True)
        has_gcovr = True
        print("Found gcovr")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        subprocess.run(["lcov", "--version"], capture_output=True, check=True)
        has_lcov = True
        print("Found lcov")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    if not has_gcovr and not has_lcov:
        print("ERROR: Neither gcovr nor lcov is installed")
        print("       Install with: sudo apt-get install gcovr lcov")
        print("       (gcovr is recommended for XML support)")
        return
    
    # Find build directory
    build_dirs = []
    for arch in ["x86_64", "aarch64"]:
        build_dir = PROJECT_ROOT / f"build_{arch}"
        if build_dir.exists():
            build_dirs.append(build_dir)
    
    if not build_dirs:
        # Fallback to generic build directory
        if BUILD_DIR.exists():
            build_dirs.append(BUILD_DIR)
    
    if not build_dirs:
        print("ERROR: No build directory found")
        print("       Expected: build_x86_64, build_aarch64, or build/")
        return
    
    # Use the first found build directory
    build_dir = build_dirs[0]
    print(f"Using build directory: {build_dir}")
    
    # Check for .gcda files
    gcda_files = list(build_dir.rglob("*.gcda"))
    if not gcda_files:
        print("ERROR: No coverage data found (.gcda files)")
        print("       Make sure you built with --coverage flag:")
        print("       ./build.sh --clean --coverage")
        return
    
    print(f"Found {len(gcda_files)} coverage data files")
    
    # Output paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coverage_dir = Path(__file__).parent / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    
    # Generate reports based on available tools
    success = False
    
    if has_gcovr:
        success = generate_gcovr_reports(build_dir, coverage_dir, timestamp)
    elif has_lcov:
        success = generate_lcov_reports(build_dir, coverage_dir, timestamp)
    
    if success:
        print("\n" + "=" * 80)
        print("Coverage reports generated successfully")
        print("=" * 80)


def generate_gcovr_reports(build_dir, coverage_dir, timestamp):
    """Generate coverage reports using gcovr (supports XML, HTML, JSON)"""
    try:
        # Output file paths
        xml_file = coverage_dir / f"coverage_{timestamp}.xml"
        html_dir = coverage_dir / "html"
        json_file = coverage_dir / f"coverage_{timestamp}.json"
        
        # Filter patterns
        exclude_patterns = [
            r".*/usr/.*",
            r".*/third_party/.*",
            r".*/extern/.*",
            r".*/tests/.*",
            r".*/test/.*",
        ]
        
        exclude_args = []
        for pattern in exclude_patterns:
            exclude_args.extend(["--exclude", pattern])
        
        # Check gcovr version to determine if it supports --gcov-ignore-parse-errors
        ignore_parse_errors_arg = []
        try:
            result = subprocess.run(["gcovr", "--version"], capture_output=True, text=True)
            # gcovr 5.0+ supports --gcov-ignore-parse-errors
            version_line = result.stdout.split('\n')[0]
            if 'gcovr' in version_line:
                # Try to extract version number
                import re
                match = re.search(r'(\d+)\.(\d+)', version_line)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    if major >= 5:
                        ignore_parse_errors_arg = ["--gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file"]
        except Exception:
            pass  # If version check fails, proceed without the option
        
        # Create HTML output directory
        html_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports (separate commands for compatibility)
        print("\nGenerating coverage reports...")
        
        # Generate XML report
        cmd = [
            "gcovr",
            "--root", str(PROJECT_ROOT),
            str(build_dir),
            "--xml", str(xml_file),
            "--xml-pretty",
        ] + ignore_parse_errors_arg + exclude_args
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to generate XML report:")
            print(result.stderr)
            return False
        
        # Generate HTML report
        cmd = [
            "gcovr",
            "--root", str(PROJECT_ROOT),
            str(build_dir),
            "--html-details", str(html_dir / "index.html"),
            "--html-title", "dx_app C++ Code Coverage",
        ] + ignore_parse_errors_arg + exclude_args
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: Failed to generate HTML report:")
            print(result.stderr)
        
        # Generate JSON report
        cmd = [
            "gcovr",
            "--root", str(PROJECT_ROOT),
            str(build_dir),
            "--json", str(json_file),
            "--json-pretty",
        ] + ignore_parse_errors_arg + exclude_args
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: Failed to generate JSON report:")
            print(result.stderr)
        
        # Print summary separately
        print("\nCoverage Summary:")
        cmd_summary = [
            "gcovr",
            "--root", str(PROJECT_ROOT),
            str(build_dir),
        ] + ignore_parse_errors_arg + exclude_args
        
        result = subprocess.run(cmd_summary, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.rstrip())
        
        # Print report locations
        print(f"\nCoverage reports generated:")
        print(f"  XML:  {xml_file}")
        print(f"  HTML: {html_dir}/index.html")
        print(f"  JSON: {json_file}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error generating coverage reports: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_lcov_reports(build_dir, coverage_dir, timestamp):
    """Generate coverage reports using lcov (HTML only)"""
    try:
        coverage_info = coverage_dir / f"coverage_{timestamp}.info"
        coverage_filtered = coverage_dir / f"coverage_{timestamp}_filtered.info"
        html_dir = coverage_dir / "html"
        
        # Step 1: Capture coverage data
        print("\nCapturing coverage data...")
        cmd = [
            "lcov",
            "--capture",
            "--directory", str(build_dir),
            "--output-file", str(coverage_info),
            "--rc", "lcov_branch_coverage=1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to capture coverage data:")
            print(result.stderr)
            return False
        print("Coverage data captured")
        
        # Step 2: Filter out unwanted files
        print("\nFiltering coverage data...")
        cmd = [
            "lcov",
            "--remove", str(coverage_info),
            "/usr/*",
            "*/third_party/*",
            "*/extern/*",
            "*/tests/*",
            "*/test/*",
            "--output-file", str(coverage_filtered),
            "--rc", "lcov_branch_coverage=1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to filter coverage data:")
            print(result.stderr)
            return False
        print("Coverage data filtered")
        
        # Step 3: Generate HTML report
        print("\nGenerating HTML report...")
        cmd = [
            "genhtml",
            str(coverage_filtered),
            "--output-directory", str(html_dir),
            "--rc", "lcov_branch_coverage=1",
            "--title", "dx_app C++ Code Coverage",
            "--legend"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to generate HTML report:")
            print(result.stderr)
            return False
        print("HTML report generated")
        
        # Step 4: Show summary
        print("\nCoverage Summary:")
        cmd = [
            "lcov",
            "--summary", str(coverage_filtered),
            "--rc", "lcov_branch_coverage=1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        # Print report location
        print(f"\nCoverage reports:")
        print(f"  LCOV: {coverage_filtered.absolute()}")
        print(f"  HTML: {html_dir.absolute()}/index.html")
        print(f"\n  Open in browser: file://{html_dir.absolute()}/index.html\n")
        print(f"\nNOTE: XML output requires gcovr. Install with: sudo apt-get install gcovr\n")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error generating coverage report: {e}")
        import traceback
        traceback.print_exc()
        return False

