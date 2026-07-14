"""
Python example script invalid argument tests.

Validates that Python inference scripts (yolov7_sync.py, yolov7_async.py)
handle invalid arguments gracefully: missing model, nonexistent files,
invalid options, etc.

Prerequisites:
    - setup.bat completed (model & video in assets/)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import run_command, PROJECT_ROOT, PYTHON_EXAMPLE_DIR, MODELS_DIR


# ======================================================================
# Discovery: find Python example scripts
# ======================================================================

def _discover_python_scripts():
    """Find all *_sync.py and *_async.py scripts in python_example."""
    if not PYTHON_EXAMPLE_DIR.exists():
        return []
    scripts = []
    for py_file in sorted(PYTHON_EXAMPLE_DIR.rglob("*.py")):
        if py_file.name.startswith("__"):
            continue
        if "_sync" in py_file.stem or "_async" in py_file.stem:
            scripts.append(py_file)
    return scripts


ALL_SCRIPTS = _discover_python_scripts()
SCRIPT_IDS = [f"{s.parent.parent.name}/{s.stem}" for s in ALL_SCRIPTS]


# ======================================================================
# Tests targeting YOLOv7 specifically (reference demo)
# ======================================================================

class TestYOLOv7PythonInvalidArgs:
    """Invalid argument tests using yolov7_sync.py as the reference."""

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_no_arguments(self, yolov7_py_sync):
        """Running with no arguments should exit non-zero (--model required)."""
        result = run_command([sys.executable, str(yolov7_py_sync)])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_unknown_flag(self, yolov7_py_sync):
        """Unrecognized option should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--unknown-flag-xyz",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_model_file_not_found(self, yolov7_py_sync, yolov7_image):
        """Non-existent model file should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", "nonexistent_model_12345.dxnn",
            "--image", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_image_file_not_found(self, yolov7_py_sync, yolov7_model):
        """Non-existent image file should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", str(yolov7_model),
            "--image", "nonexistent_image_12345.jpg",
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_video_file_not_found(self, yolov7_py_sync, yolov7_model):
        """Non-existent video file should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", str(yolov7_model),
            "--video", "nonexistent_video_12345.mp4",
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_empty_model_file(self, yolov7_py_sync, yolov7_image):
        """Empty model file (0 bytes) should exit non-zero."""
        with tempfile.NamedTemporaryFile(suffix=".dxnn", delete=False) as f:
            empty_model = f.name

        try:
            result = run_command([
                sys.executable, str(yolov7_py_sync),
                "--model", empty_model,
                "--image", str(yolov7_image),
                "--no-display",
            ])
            assert result.returncode != 0
        finally:
            Path(empty_model).unlink(missing_ok=True)

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_corrupted_model_file(self, yolov7_py_sync, yolov7_image):
        """Corrupted model file should exit non-zero."""
        with tempfile.NamedTemporaryFile(suffix=".dxnn", delete=False) as f:
            f.write(b"CORRUPTED_NOT_A_REAL_MODEL" * 50)
            corrupted_model = f.name

        try:
            result = run_command([
                sys.executable, str(yolov7_py_sync),
                "--model", corrupted_model,
                "--image", str(yolov7_image),
                "--no-display",
            ])
            assert result.returncode != 0
        finally:
            Path(corrupted_model).unlink(missing_ok=True)

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_model_flag_without_value(self, yolov7_py_sync):
        """--model without a value should exit non-zero."""
        result = run_command([sys.executable, str(yolov7_py_sync), "--model"])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_image_flag_without_value(self, yolov7_py_sync, yolov7_model):
        """--image without a value should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", str(yolov7_model),
            "--image",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_video_flag_without_value(self, yolov7_py_sync, yolov7_model):
        """--video without a value should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", str(yolov7_model),
            "--video",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_wrong_file_as_model(self, yolov7_py_sync, yolov7_image):
        """Using an image file as model should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", str(yolov7_image),
            "--image", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_directory_as_model(self, yolov7_py_sync, yolov7_image):
        """Using a directory path as model should exit non-zero."""
        result = run_command([
            sys.executable, str(yolov7_py_sync),
            "--model", str(PROJECT_ROOT / "assets"),
            "--image", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0


# ======================================================================
# Tests targeting async variant
# ======================================================================

class TestYOLOv7PythonAsyncInvalidArgs:
    """Same invalid argument tests for yolov7_async.py."""

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_no_arguments(self, yolov7_py_async):
        result = run_command([sys.executable, str(yolov7_py_async)])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_unknown_flag(self, yolov7_py_async):
        result = run_command([
            sys.executable, str(yolov7_py_async), "--invalid-flag-xyz",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_model_file_not_found(self, yolov7_py_async, yolov7_image):
        result = run_command([
            sys.executable, str(yolov7_py_async),
            "--model", "fake_model.dxnn",
            "--image", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.python_script
    @pytest.mark.invalid_args
    def test_video_file_not_found(self, yolov7_py_async, yolov7_model):
        result = run_command([
            sys.executable, str(yolov7_py_async),
            "--model", str(yolov7_model),
            "--video", "fake_video.mp4",
            "--no-display",
        ])
        assert result.returncode != 0


# ======================================================================
# Parametrized tests across ALL discovered scripts
# ======================================================================

@pytest.mark.python_script
@pytest.mark.invalid_args
@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=SCRIPT_IDS)
def test_all_scripts_no_args(script):
    """All Python scripts should fail with no arguments (--model is required)."""
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    result = run_command([sys.executable, str(script)])
    assert result.returncode != 0, (
        f"{script.name} should require --model but returned 0"
    )


@pytest.mark.python_script
@pytest.mark.invalid_args
@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=SCRIPT_IDS)
def test_all_scripts_invalid_flag(script):
    """All Python scripts should reject unrecognized options."""
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    result = run_command([sys.executable, str(script), "--nonexistent-option-zzz"])
    assert result.returncode != 0, (
        f"{script.name} accepted an invalid flag"
    )


@pytest.mark.python_script
@pytest.mark.invalid_args
@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=SCRIPT_IDS)
def test_all_scripts_nonexistent_model(script):
    """All Python scripts should fail gracefully with a nonexistent model path."""
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    result = run_command([
        sys.executable, str(script),
        "--model", "totally_fake_model_XXXX.dxnn",
        "--image", "sample/img/sample_street.jpg",
        "--no-display",
    ])
    assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
