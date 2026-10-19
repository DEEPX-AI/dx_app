"""
Basic CLI tests for bin executables
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.proc import BoundedTimeout, run_bounded  # noqa: E402
from test_helpers.utils import binary_path, setup_environment  # noqa: E402

from conftest import UNIT_TEST_BINARIES, is_executable, resolve_bin_dir


PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"


def get_executables():
    """Get list of all executable files in bin directory"""
    if not BIN_DIR.exists():
        return []
    
    executables = []
    for file in BIN_DIR.iterdir():
        if is_executable(file) and file.name not in UNIT_TEST_BINARIES:
            executables.append(file.name)
    
    return sorted(executables)


EXECUTABLES = get_executables()


@pytest.mark.cli
@pytest.mark.parametrize("executable", EXECUTABLES)
def test_invalid_arguments(executable, bin_dir):
    """
    Test that executables handle invalid arguments gracefully
    """
    executable_path = binary_path(bin_dir, executable)
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    # Skip demo_multi_channel as it has different argument handling
    if Path(executable).stem == "demo_multi_channel":
        pytest.skip("demo_multi_channel has different argument handling")
    
    env = setup_environment()
    
    try:
        result = run_bounded(
            [str(executable_path), "--invalid-option-that-does-not-exist"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env
        )
        
        # Should exit with non-zero code for invalid option
        assert result.returncode != 0, (
            f"{executable} should fail with invalid option but returned {result.returncode}"
        )
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} with invalid option timed out")
    except Exception as e:
        pytest.fail(f"{executable} with invalid option raised exception: {e}")


@pytest.mark.cli
@pytest.mark.parametrize("executable", EXECUTABLES)
def test_no_arguments(executable, bin_dir):
    """
    Test that executables handle no arguments appropriately
    
    Most should either show help or exit with error
    """
    executable_path = binary_path(bin_dir, executable)
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    # Skip demo_multi_channel as it might behave differently
    if Path(executable).stem == "demo_multi_channel":
        pytest.skip("demo_multi_channel has different behavior")
    
    env = setup_environment()
    
    try:
        result = run_bounded(
            [str(executable_path)],
            capture_output=True,
            text=True,
            timeout=15,
            env=env
        )
        returncode = result.returncode
    except BoundedTimeout as e:
        # With no arguments a runner whose default model is present runs its
        # default demo, and in image mode it then keeps the window open until
        # someone closes it. Headless (offscreen) nobody can, so the timeout's
        # SIGTERM stands in for closing it. That run must have really run and
        # wound down: exit 0 with its interrupt line or its summary. (U-47:
        # this branch used to accept 0/1/2/255, so a runner that hung before
        # doing anything and then exited 0 on SIGTERM passed too.)
        if e.killed:
            pytest.fail(f"{executable} with no arguments ignored SIGTERM: {e}")
        text = (e.output or "") + (e.stderr or "")
        assert e.returncode == 0, (
            f"{executable} with no arguments: rc={e.returncode} after SIGTERM\n{text[-1500:]}")
        assert "Interrupted by user" in text or "PERFORMANCE SUMMARY" in text, (
            f"{executable} with no arguments timed out without running its demo:\n{text[-1500:]}")
        return
    except Exception as e:
        pytest.fail(f"{executable} with no arguments raised exception: {e}")

    # Most apps should exit with non-zero when required args are missing
    # We just check it doesn't crash
    assert returncode in [0, 1, 2, 255], (
        f"{executable} with no args returned unexpected code {returncode}"
    )


DEFAULT_MODEL_EXAMPLE = "yolov8-n_640x640_sync"
DEFAULT_MODEL_PATH = "assets/models/yolov8-n_640x640.dxnn"


@pytest.mark.e2e
def test_example_without_model_option_runs_its_default_model(bin_dir):
    """SDKREQ-529: with -m omitted the example loads its own variant's model
    from the registry. Skipped when the binary or the model is absent (a
    missing model would start the auto-downloader)."""
    executable_path = binary_path(bin_dir, DEFAULT_MODEL_EXAMPLE)
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    if not (PROJECT_ROOT / DEFAULT_MODEL_PATH).is_file():
        pytest.skip(f"{DEFAULT_MODEL_PATH} not downloaded")

    result = run_bounded(
        [str(executable_path), "-i", "sample/img/sample_dog.jpg", "--no-display"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
        env=setup_environment())
    text = result.stdout + result.stderr
    assert result.returncode == 0, text[-2000:]
    assert "Using example default: " + DEFAULT_MODEL_PATH in text, text[-2000:]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
