"""
Basic CLI tests for Python inference scripts.

Tests invalid arguments, missing required args, and import-based argument
parsing (--image, --video, --camera, --rtsp, --display / --no-display)
without running actual inference.

Mirrors ``tests/cpp_example/test_cli_basic.py``.
"""
import subprocess
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import PROJECT_ROOT  # noqa: E402
from test_helpers.utils import discover_python_scripts, setup_environment  # noqa: E402
from conftest import load_module_from_file  # noqa: E402


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_all_scripts() -> List[pytest.param]:
    raw = discover_python_scripts(suffixes=("_sync", "_async"))
    params = []
    for _task, _model_name, sync_scripts, async_scripts, _model in raw:
        for script, mode in ([(s, "sync") for s in sync_scripts] + [(s, "async") for s in async_scripts]):
            marker = pytest.mark.sync_exec if mode == "sync" else pytest.mark.async_exec
            params.append(pytest.param(script, id=script.stem, marks=marker))
    return params


def _discover_sync_scripts() -> List[pytest.param]:
    raw = discover_python_scripts(suffixes=("_sync",))
    params = []
    for _task, _model_name, sync_scripts, _async_scripts, _model in raw:
        for sync_script in sync_scripts:
            params.append(pytest.param(sync_script, id=sync_script.stem, marks=pytest.mark.sync_exec))
    return params


SCRIPT_PARAMS = _discover_all_scripts()
SYNC_SCRIPT_PARAMS = _discover_sync_scripts()


def _run(cmd: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=setup_environment(),
        cwd=str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# Subprocess-based tests
# ---------------------------------------------------------------------------

@pytest.mark.cli
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_missing_required_args(script: Path):
    """Running with no arguments should exit non-zero (--model is required)."""
    try:
        result = _run([sys.executable, str(script)])
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} (no args) timed out")

    assert result.returncode != 0, (
        f"{script.name} should fail with no arguments but returned {result.returncode}"
    )


@pytest.mark.cli
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_invalid_arguments(script: Path):
    """Unrecognised option should exit non-zero."""
    try:
        result = _run([sys.executable, str(script), "--invalid-option-that-does-not-exist"])
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} (invalid opt) timed out")

    assert result.returncode != 0, (
        f"{script.name} should fail with invalid option but returned {result.returncode}"
    )


# ---------------------------------------------------------------------------
# Import-based argument parsing tests
# ---------------------------------------------------------------------------

def _get_parse_fn(script: Path):
    """Load script module and return its parse_arguments / parse_args function."""
    module = load_module_from_file(str(script), script.stem)
    if module is None:
        pytest.skip(f"Failed to load module {script.name}")
    parse_fn = getattr(module, "parse_arguments", None) or getattr(module, "parse_args", None)
    if parse_fn is None:
        pytest.skip(f"No argument parser found in {script.name}")
    return parse_fn


@pytest.mark.cli
@pytest.mark.parametrize("script", SYNC_SCRIPT_PARAMS)
def test_cli_image_mode(script: Path):
    """--model + --image should be accepted and parsed correctly."""
    parse_fn = _get_parse_fn(script)
    with patch("sys.argv", [str(script), "--model", "test.dxnn", "--image", "test.jpg"]):
        with patch("os.path.exists", return_value=True):
            args = parse_fn()
    assert hasattr(args, "model") and args.model == "test.dxnn"
    assert hasattr(args, "image") and args.image == "test.jpg"


@pytest.mark.cli
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_cli_video_mode(script: Path):
    """--model + --video should be accepted and parsed correctly."""
    parse_fn = _get_parse_fn(script)
    with patch("sys.argv", [str(script), "--model", "test.dxnn", "--video", "test.mp4"]):
        with patch("os.path.exists", return_value=True):
            args = parse_fn()
    assert hasattr(args, "model") and args.model == "test.dxnn"
    assert hasattr(args, "video") and args.video == "test.mp4"


@pytest.mark.cli
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_cli_display_options(script: Path):
    """--display (default) and --no-display should parse correctly."""
    parse_fn = _get_parse_fn(script)
    with patch("sys.argv", [str(script), "--model", "test.dxnn", "--video", "test.mp4"]):
        with patch("os.path.exists", return_value=True):
            args = parse_fn()
    assert hasattr(args, "display") and args.display is True

    with patch("sys.argv", [str(script), "--model", "test.dxnn", "--video", "test.mp4", "--no-display"]):
        with patch("os.path.exists", return_value=True):
            args = parse_fn()
    assert args.display is False


@pytest.mark.cli
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_cli_camera_mode(script: Path):
    """--camera 0 should be accepted (skip if not supported by the script)."""
    parse_fn = _get_parse_fn(script)
    with patch("sys.argv", [str(script), "--model", "test.dxnn", "--camera", "0"]):
        with patch("os.path.exists", return_value=True):
            try:
                with patch("sys.stderr"):
                    args = parse_fn()
            except SystemExit as e:
                if e.code == 2:
                    pytest.skip(f"{script.name}: --camera not supported")
                raise
    assert hasattr(args, "model") and args.model == "test.dxnn"
    assert hasattr(args, "camera") and args.camera == 0


@pytest.mark.cli
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_cli_rtsp_mode(script: Path):
    """--rtsp URL should be accepted (skip if not supported by the script)."""
    parse_fn = _get_parse_fn(script)
    rtsp_url = "rtsp://fake.url/stream"
    with patch("sys.argv", [str(script), "--model", "test.dxnn", "--rtsp", rtsp_url]):
        with patch("os.path.exists", return_value=True):
            try:
                with patch("sys.stderr"):
                    args = parse_fn()
            except SystemExit as e:
                if e.code == 2:
                    pytest.skip(f"{script.name}: --rtsp not supported")
                raise
    assert hasattr(args, "model") and args.model == "test.dxnn"
    assert hasattr(args, "rtsp") and args.rtsp == rtsp_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
