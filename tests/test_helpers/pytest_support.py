"""Shared pytest wiring for the cpp_example / python_example suites.

Both suites need the same CLI options (``--loop``, ``--camera-index``,
``--rtsp-url``, ``--stream-duration``) and the same stream/thermal fixtures.
They used to carry two independent copies of this code, which drifted. This
module is the single source; each ``conftest.py`` re-exports what it needs.

CONTRACT -- do not rename anything here without updating ``tests/req_test``.
``tests/req_test`` verifies each SDKREQ requirement at runtime by driving these
suites through ``pytest -m <marker>`` and through the ``--camera-index`` /
``--rtsp-url`` options, so the option names, fixture names and default values
below are part of that contract. See ``tests/README.md``, section
"req_test contract surface".
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

#: ``dx_app/tests``
TESTS_DIR = Path(__file__).resolve().parent.parent
#: ``dx_app``
PROJECT_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

#: Temperature (Celsius) the NPU must fall below before an ``e2e`` test starts.
E2E_TARGET_TEMP_C = 70

logger = logging.getLogger(__name__)


def configure_stdio_errors() -> None:
    """Keep stdout/stderr alive when an example prints non-UTF8 bytes.

    Example binaries occasionally emit locale-dependent bytes. Without this the
    captured stream raises UnicodeDecodeError and takes the whole run down.
    """
    for stream in (sys.__stdout__, sys.__stderr__, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            pass  # None (pythonw), or a capture object without reconfigure()


def _addoption_once(parser, *names, **attrs) -> None:
    """``parser.addoption`` that tolerates a second conftest having added it already.

    Both suites register the shared options, so a session that loads both
    conftests (``pytest tests/cpp_example/... tests/python_example/...``) sees
    each option twice; the first registration wins.
    """
    try:
        parser.addoption(*names, **attrs)
    except ValueError as exc:
        msg = str(exc)
        if "already added" not in msg and "conflicting" not in msg:
            raise


def add_loop_option(parser, default: str) -> None:
    """Register ``--loop``; tolerate a second conftest having added it already.

    The two suites use different defaults (Python: 1, C++ image: 20), so the default
    is a parameter rather than a constant.
    """
    _addoption_once(
        parser,
        "--loop",
        action="store",
        default=default,
        help=(
            "Number of inference iterations for E2E image tests "
            f"(default: {default})"
        ),
    )


def add_stream_options(parser) -> None:
    """Register the camera / RTSP options shared by both suites (once per session)."""
    _addoption_once(
        parser,
        "--camera-index",
        action="store",
        default=None,
        help="Camera device index for e2e_camera tests (e.g. 0)",
    )
    _addoption_once(
        parser,
        "--rtsp-url",
        action="store",
        default=None,
        help="RTSP stream URL for e2e_rtsp tests",
    )
    _addoption_once(
        parser,
        "--stream-duration",
        action="store",
        default="10",
        help="Seconds to run each camera/RTSP test (default: 10)",
    )


@pytest.fixture(scope="session")
def camera_index(request):
    """Camera device index from --camera-index."""
    val = request.config.getoption("--camera-index")
    if val is None:
        pytest.skip("--camera-index not provided")
    return int(val)


@pytest.fixture(scope="session")
def rtsp_url(request):
    """RTSP URL from --rtsp-url."""
    val = request.config.getoption("--rtsp-url")
    if val is None:
        pytest.skip("--rtsp-url not provided")
    return val


@pytest.fixture(scope="session")
def stream_duration(request) -> int:
    """Seconds to run each camera/RTSP test."""
    try:
        return int(request.config.getoption("--stream-duration"))
    except (TypeError, ValueError):
        return 10


@pytest.fixture(scope="function", autouse=True)
def wait_for_temperature(request):
    """Let the NPU cool below ``E2E_TARGET_TEMP_C`` before each ``e2e`` test.

    A no-op for every non-``e2e`` test, and on hosts that have neither ``bash``
    nor ``scripts/check_temperature.sh``.
    """
    if not request.node.get_closest_marker("e2e"):
        return
    check_temp_script = SCRIPTS_DIR / "check_temperature.sh"
    # bash is absent on stock Windows: resolve it instead of assuming PATH,
    # or every e2e test dies at setup with FileNotFoundError (WinError 2).
    bash = shutil.which("bash")
    if not (bash and check_temp_script.exists()):
        return
    result = subprocess.run(
        [bash, str(check_temp_script), f"--wait_target_temp={E2E_TARGET_TEMP_C}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if "Waiting" in result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info(line.strip())
