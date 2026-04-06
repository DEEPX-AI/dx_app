"""
Multi-loop tests for Python inference scripts.

Verifies that ``--loop N`` processes multiple iterations without crashing.
Tests both sync and async scripts with a small loop count on video input.

Mirrors ``tests/cpp_example/test_multi_loop.py``.
"""

import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.constants import PROJECT_ROOT  # noqa: E402
from common.utils import discover_python_scripts, setup_environment  # noqa: E402


_TEST_VIDEO = PROJECT_ROOT / "assets" / "videos" / "dance-group.mov"
_LOOP_COUNT = 3


# ---------------------------------------------------------------------------
# Discovery — sync + async scripts with a model
# ---------------------------------------------------------------------------

def _discover_scripts() -> List[pytest.param]:
    raw = discover_python_scripts(suffixes=("_sync", "_async"))
    params = []
    for _task, _model_name, sync_scripts, async_scripts, model_path in raw:
        if model_path is None:
            continue
        for script, mode in ([(s, "sync") for s in sync_scripts] + [(s, "async") for s in async_scripts]):
            marker = pytest.mark.sync_exec if mode == "sync" else pytest.mark.async_exec
            params.append(pytest.param(
                script, model_path,
                id=script.stem,
                marks=marker,
            ))
    return params


SCRIPT_PARAMS = _discover_scripts()

if not SCRIPT_PARAMS:
    pytest.skip(
        "No models found in assets/models/ — run setup_sample_models.sh first",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.multi_loop
@pytest.mark.parametrize("script,model", SCRIPT_PARAMS)
def test_multi_loop_video(script: Path, model: Path):
    """Run script with --loop N on video; should complete without error."""
    if not _TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {_TEST_VIDEO}")

    cmd = [
        sys.executable, str(script),
        "--model", str(model),
        "--video", str(_TEST_VIDEO),
        "--loop", str(_LOOP_COUNT),
        "--no-display",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=setup_environment(),
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} --loop {_LOOP_COUNT} timed out")

    assert result.returncode == 0, (
        f"{script.name}: multi-loop FAILED (exit {result.returncode})\n"
        f"CMD    : {' '.join(cmd)}\n"
        f"stdout : {result.stdout[-2000:]}\nstderr : {result.stderr[-2000:]}"
    )
