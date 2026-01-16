"""
Test --help option for all executables in bin directory
"""
import os
import subprocess
from pathlib import Path

import pytest

from conftest import is_executable, resolve_bin_dir


# Get all executables from bin directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"


def get_executables():
    """Get list of all executable files in bin directory"""
    if not BIN_DIR.exists():
        return []
    
    executables = []
    for file in BIN_DIR.iterdir():
        if file.is_file() and file.stat().st_mode & 0o111:  # Check if executable
            executables.append(file.name)
    
    return sorted(executables)


EXECUTABLES = get_executables()


def build_environment():
    env = os.environ.copy()
    if LIB_DIR.exists():
        if os.name == "nt":
            current_path = env.get("PATH", "")
            env["PATH"] = f"{LIB_DIR};{current_path}" if current_path else str(LIB_DIR)
        else:
            current_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{current_ld_path}" if current_ld_path else str(LIB_DIR)
    return env


@pytest.mark.cli
@pytest.mark.help
@pytest.mark.parametrize("executable", EXECUTABLES)
def test_help_option(executable, bin_dir):
    """
    Test that each executable responds to --help without errors
    
    Args:
        executable: Name of the executable to test
        bin_dir: Path to bin directory (from fixture)
    """
    executable_path = bin_dir / executable
    
    # Skip if executable doesn't exist
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    # Setup environment with library path
    env = build_environment()
    
    # Run executable with --help
    try:
        result = subprocess.run(
            [str(executable_path), "--help"],
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
            env=env
        )
        
        # Check that it exits successfully (usually 0 or sometimes 1 for help)
        assert result.returncode in [0, 1], (
            f"{executable} --help failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        
        # Check that some output was produced (either stdout or stderr)
        output = result.stdout + result.stderr
        assert len(output) > 0, (
            f"{executable} --help produced no output"
        )
        
        # Check for common help indicators
        output_lower = output.lower()
        has_help_content = any(
            keyword in output_lower
            for keyword in ["usage", "options", "arguments", "help", "example"]
        )
        
        assert has_help_content, (
            f"{executable} --help output doesn't contain expected help keywords\n"
            f"Output: {output[:500]}"
        )
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} --help timed out after 5 seconds")
    except Exception as e:
        pytest.fail(f"{executable} --help raised exception: {e}")


@pytest.mark.cli
@pytest.mark.help
def test_all_executables_found(bin_dir):
    """
    Sanity check that we found executables in bin directory
    """
    executables = get_executables()
    assert len(executables) > 0, f"No executables found in {bin_dir}"
    print(f"\nFound {len(executables)} executables:")
    for exe in executables:
        print(f"  - {exe}")


@pytest.mark.cli
@pytest.mark.help
@pytest.mark.cli
@pytest.mark.help
@pytest.mark.parametrize("executable", EXECUTABLES)
def test_help_option_shows_usage(executable, bin_dir):
    """
    Test that --help shows usage information
    
    More specific test checking for "Usage:" or similar patterns
    """
    executable_path = bin_dir / executable
    
    if not executable_path.exists():
        pytest.skip(f"Executable not found: {executable_path}")
    
    # Setup environment with library path
    env = build_environment()
    
    try:
        result = subprocess.run(
            [str(executable_path), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env
        )
        
        output = result.stdout + result.stderr
        output_lower = output.lower()
        
        # Check for usage pattern
        has_usage = "usage" in output_lower or executable in output
        
        assert has_usage, (
            f"{executable} --help doesn't show usage information\n"
            f"Output: {output[:500]}"
        )
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"{executable} --help timed out")
    except Exception as e:
        pytest.fail(f"{executable} --help raised exception: {e}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v"])
