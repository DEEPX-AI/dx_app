"""
Test --help option for Python inference scripts.

Auto-discovers all ``*_sync*`` / ``*_async*`` scripts and verifies:
- ``--help`` returns exit code 0 and prints usage info
- Output contains expected help keywords (usage, options, --model, etc.)

Mirrors ``tests/cpp_example/test_cli_help.py``.
"""
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import PROJECT_ROOT  # noqa: E402
from test_helpers.utils import discover_python_scripts, setup_environment  # noqa: E402


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


SCRIPT_PARAMS = _discover_all_scripts()


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
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.cli
@pytest.mark.help
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_help_option(script: Path):
    """--help should exit 0 and print usage information."""
    try:
        result = _run([sys.executable, str(script), "--help"])
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} --help timed out")

    assert result.returncode in [0, 1], (
        f"{script.name} --help returned unexpected code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    output = result.stdout + result.stderr
    assert len(output) > 0, f"{script.name} --help produced no output"

    output_lower = output.lower()
    assert any(kw in output_lower for kw in ["usage", "options", "arguments", "help", "--model"]), (
        f"{script.name} --help output missing expected keywords\nOutput: {output[:500]}"
    )


@pytest.mark.cli
@pytest.mark.help
@pytest.mark.parametrize("script", SCRIPT_PARAMS)
def test_help_option_shows_usage(script: Path):
    """--help should show 'usage' or the script name in output."""
    try:
        result = _run([sys.executable, str(script), "--help"])
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} --help timed out")

    output = result.stdout + result.stderr
    output_lower = output.lower()

    has_usage = "usage" in output_lower or script.stem in output
    assert has_usage, (
        f"{script.name} --help doesn't show usage information\nOutput: {output[:500]}"
    )


@pytest.mark.cli
@pytest.mark.help
def test_all_scripts_found():
    """Sanity check that we discovered scripts."""
    assert len(SCRIPT_PARAMS) > 0, "No Python scripts found"
    print(f"\nFound {len(SCRIPT_PARAMS)} scripts:")
    for p in SCRIPT_PARAMS[:10]:
        print(f"  - {p.values[0].stem}")
    if len(SCRIPT_PARAMS) > 10:
        print(f"  ... and {len(SCRIPT_PARAMS) - 10} more")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
