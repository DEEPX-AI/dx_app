"""
Basic CLI tests for bin executables
"""
import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
LIB_DIR = PROJECT_ROOT / "lib"


def get_executables():
    """Get list of all executable files in bin directory"""
    if not BIN_DIR.exists():
        return []
    
    executables = []
    for file in BIN_DIR.iterdir():
        if file.is_file() and file.stat().st_mode & 0o111:
            executables.append(file.name)
    
    return sorted(executables)


EXECUTABLES = get_executables()


def setup_environment():
    """Setup environment with required library paths"""
    env = os.environ.copy()
    if LIB_DIR.exists():
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        if current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{current_ld_path}"
        else:
            env["LD_LIBRARY_PATH"] = str(LIB_DIR)
    return env


@pytest.mark.cli
@pytest.mark.parametrize("executable", EXECUTABLES)
def test_invalid_arguments(executable, bin_dir):
    """
    Test that executables handle invalid arguments gracefully
    """
    executable_path = bin_dir / executable
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    # Skip demo_multi_channel as it has different argument handling
    if executable == "demo_multi_channel":
        pytest.skip("demo_multi_channel has different argument handling")
    
    env = setup_environment()
    
    try:
        result = subprocess.run(
            [str(executable_path), "--invalid-option-that-does-not-exist"],
            capture_output=True,
            text=True,
            timeout=5,
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
    executable_path = bin_dir / executable
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    # Skip demo_multi_channel as it might behave differently
    if executable == "demo_multi_channel":
        pytest.skip("demo_multi_channel has different behavior")
    
    env = setup_environment()
    
    try:
        result = subprocess.run(
            [str(executable_path)],
            capture_output=True,
            text=True,
            timeout=5,
            env=env
        )
        
        # Most apps should exit with non-zero when required args are missing
        # We just check it doesn't crash
        assert result.returncode in [0, 1, 2, 255], (
            f"{executable} with no args returned unexpected code {result.returncode}"
        )
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} with no arguments timed out")
    except Exception as e:
        pytest.fail(f"{executable} with no arguments raised exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
