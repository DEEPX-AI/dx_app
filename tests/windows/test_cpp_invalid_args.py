"""
C++ executable invalid argument tests.

Validates that C++ binaries (yolov7_sync, yolov7_async) handle invalid
arguments gracefully: wrong flags, missing files, corrupted model, etc.

Prerequisites:
    - build.bat completed (executables in bin/Release/)
    - setup.bat completed (model & video in assets/)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import run_command, BIN_DIR, MODELS_DIR, PROJECT_ROOT


# ======================================================================
# Discovery: find all *_sync.exe and *_async.exe in bin dir
# ======================================================================

def _discover_executables():
    """Find executables ending with _sync.exe or _async.exe."""
    if not BIN_DIR.exists():
        return []
    exes = []
    for f in sorted(BIN_DIR.iterdir()):
        if f.suffix.lower() == ".exe" and (f.stem.endswith("_sync") or f.stem.endswith("_async")):
            exes.append(f)
    return exes


ALL_EXECUTABLES = _discover_executables()
EXE_IDS = [e.stem for e in ALL_EXECUTABLES]


# ======================================================================
# Tests targeting YOLOv7 specifically (reference demo)
# ======================================================================

class TestYOLOv7CppInvalidArgs:
    """Invalid argument tests using yolov7_sync as the reference."""

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_no_arguments(self, yolov7_cpp_sync):
        """No arguments is a VALID run now (SDKREQ-529): -m omitted resolves the
        example default model and no input falls back to the default sample.
        So we only require it to be handled gracefully, not to fail."""
        result = run_command([str(yolov7_cpp_sync)])
        assert result.returncode in [0, 1, 2, 255], (
            f"Expected graceful handling for no args, got {result.returncode}"
        )

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_unknown_flag(self, yolov7_cpp_sync):
        """Unrecognized flag should exit non-zero."""
        result = run_command([str(yolov7_cpp_sync), "--unknown-flag-xyz"])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_model_file_not_found(self, yolov7_cpp_sync, yolov7_image):
        """Non-existent model file should exit non-zero."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", "nonexistent_model_12345.dxnn",
            "-i", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_image_file_not_found(self, yolov7_cpp_sync, yolov7_model):
        """Non-existent image file should exit non-zero."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", str(yolov7_model),
            "-i", "nonexistent_image_12345.jpg",
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_video_file_not_found(self, yolov7_cpp_sync, yolov7_model):
        """Non-existent video file should exit non-zero."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", str(yolov7_model),
            "-v", "nonexistent_video_12345.mp4",
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_empty_model_file(self, yolov7_cpp_sync, yolov7_image):
        """Empty model file (0 bytes) should exit non-zero."""
        with tempfile.NamedTemporaryFile(suffix=".dxnn", delete=False) as f:
            empty_model = f.name

        try:
            result = run_command([
                str(yolov7_cpp_sync),
                "-m", empty_model,
                "-i", str(yolov7_image),
                "--no-display",
            ])
            assert result.returncode != 0
        finally:
            Path(empty_model).unlink(missing_ok=True)

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_corrupted_model_file(self, yolov7_cpp_sync, yolov7_image):
        """Corrupted model file (random bytes) should exit non-zero."""
        with tempfile.NamedTemporaryFile(suffix=".dxnn", delete=False) as f:
            f.write(b"THIS_IS_NOT_A_VALID_DXNN_MODEL_FILE" * 100)
            corrupted_model = f.name

        try:
            result = run_command([
                str(yolov7_cpp_sync),
                "-m", corrupted_model,
                "-i", str(yolov7_image),
                "--no-display",
            ])
            assert result.returncode != 0
        finally:
            Path(corrupted_model).unlink(missing_ok=True)

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_model_flag_without_value(self, yolov7_cpp_sync):
        """'-m' without a following path should exit non-zero."""
        result = run_command([str(yolov7_cpp_sync), "-m"])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_image_flag_without_value(self, yolov7_cpp_sync, yolov7_model):
        """'-i' without a following path should exit non-zero."""
        result = run_command([str(yolov7_cpp_sync), "-m", str(yolov7_model), "-i"])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_video_flag_without_value(self, yolov7_cpp_sync, yolov7_model):
        """'-v' without a following path should exit non-zero."""
        result = run_command([str(yolov7_cpp_sync), "-m", str(yolov7_model), "-v"])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_invalid_loop_count(self, yolov7_cpp_sync, yolov7_model, yolov7_image):
        """Non-numeric loop count should exit non-zero."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", str(yolov7_model),
            "-i", str(yolov7_image),
            "-l", "not_a_number",
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_negative_loop_count(self, yolov7_cpp_sync, yolov7_model, yolov7_image):
        """Negative loop count should exit non-zero."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", str(yolov7_model),
            "-i", str(yolov7_image),
            "-l", "-1",
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_conflicting_input_flags(self, yolov7_cpp_sync, yolov7_model, yolov7_image, yolov7_video):
        """Both -i and -v provided simultaneously — should either fail or use last."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", str(yolov7_model),
            "-i", str(yolov7_image),
            "-v", str(yolov7_video),
            "--no-display",
            "-l", "1",
        ])
        # Either succeed (using last) or fail gracefully — must not crash
        assert result.returncode in [0, 1, 2, 255]

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_wrong_file_extension_as_model(self, yolov7_cpp_sync, yolov7_image):
        """Using an image file as model should exit non-zero."""
        result = run_command([
            str(yolov7_cpp_sync),
            "-m", str(yolov7_image),
            "-i", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0


# ======================================================================
# Tests targeting async variant
# ======================================================================

class TestYOLOv7CppAsyncInvalidArgs:
    """Same invalid argument tests for yolov7_async."""

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_no_arguments(self, yolov7_cpp_async):
        # No arguments is a valid default run now (SDKREQ-529) — handle gracefully.
        result = run_command([str(yolov7_cpp_async)])
        assert result.returncode in [0, 1, 2, 255]

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_unknown_flag(self, yolov7_cpp_async):
        result = run_command([str(yolov7_cpp_async), "--invalid-flag-xyz"])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_model_file_not_found(self, yolov7_cpp_async, yolov7_image):
        result = run_command([
            str(yolov7_cpp_async),
            "-m", "nonexistent_model.dxnn",
            "-i", str(yolov7_image),
            "--no-display",
        ])
        assert result.returncode != 0

    @pytest.mark.cpp
    @pytest.mark.invalid_args
    def test_video_file_not_found(self, yolov7_cpp_async, yolov7_model):
        result = run_command([
            str(yolov7_cpp_async),
            "-m", str(yolov7_model),
            "-v", "nonexistent_video.mp4",
            "--no-display",
        ])
        assert result.returncode != 0


# ======================================================================
# Parametrized tests across ALL discovered executables
# ======================================================================

@pytest.mark.cpp
@pytest.mark.invalid_args
@pytest.mark.parametrize("exe", ALL_EXECUTABLES, ids=EXE_IDS)
def test_all_executables_no_args(exe):
    """All executables should handle no arguments without crashing."""
    if not exe.exists():
        pytest.skip(f"Executable not found: {exe}")
    result = run_command([str(exe)])
    assert result.returncode in [0, 1, 2, 255], (
        f"{exe.name} returned unexpected code {result.returncode}"
    )


@pytest.mark.cpp
@pytest.mark.invalid_args
@pytest.mark.parametrize("exe", ALL_EXECUTABLES, ids=EXE_IDS)
def test_all_executables_invalid_flag(exe):
    """All executables should handle invalid flags without crashing."""
    if not exe.exists():
        pytest.skip(f"Executable not found: {exe}")
    result = run_command([str(exe), "--this-flag-does-not-exist"])
    assert result.returncode != 0, (
        f"{exe.name} accepted an invalid flag (returncode=0)"
    )


@pytest.mark.cpp
@pytest.mark.invalid_args
@pytest.mark.parametrize("exe", ALL_EXECUTABLES, ids=EXE_IDS)
def test_all_executables_help(exe):
    """All executables should respond to -h or --help without crashing."""
    if not exe.exists():
        pytest.skip(f"Executable not found: {exe}")
    result = run_command([str(exe), "-h"])
    # Help should exit 0 or 1
    assert result.returncode in [0, 1, 2], (
        f"{exe.name} -h returned unexpected code {result.returncode}"
    )


@pytest.mark.cpp
@pytest.mark.invalid_args
@pytest.mark.parametrize("exe", ALL_EXECUTABLES, ids=EXE_IDS)
def test_all_executables_nonexistent_model(exe):
    """All executables should fail gracefully with nonexistent model."""
    if not exe.exists():
        pytest.skip(f"Executable not found: {exe}")
    result = run_command([
        str(exe),
        "-m", "completely_fake_model_path_XXXX.dxnn",
        "-i", "sample/img/sample_street.jpg",
        "--no-display",
    ])
    assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
