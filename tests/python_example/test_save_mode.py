"""
Save-mode tests for Python inference scripts.

Verifies that ``--save`` causes the script to write output image(s) to disk.
Only sync scripts are tested (async ``image_inference`` not applicable).

Mirrors ``tests/cpp_example/test_save_mode.py``.
"""

import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import PROJECT_ROOT, TASK_IMAGE_MAP, MODEL_IMAGE_OVERRIDE  # noqa: E402
from test_helpers.utils import discover_python_scripts, setup_environment, resolve_image_for_model  # noqa: E402


# ---------------------------------------------------------------------------
# Discovery — sync scripts only
# ---------------------------------------------------------------------------

def _discover_sync_scripts() -> List[pytest.param]:
    raw = discover_python_scripts(suffixes=("_sync",))
    params = []
    for task, model_name, sync_scripts, _async, model_path in raw:
        if not sync_scripts or model_path is None:
            continue
        img_rel = resolve_image_for_model(model_name, task)
        if img_rel is None:
            img_rel = TASK_IMAGE_MAP.get(task, "sample/img/sample_kitchen.jpg")
        image_path = PROJECT_ROOT / img_rel
        for sync_script in sync_scripts:
            params.append(pytest.param(
                sync_script, model_path, image_path,
                id=sync_script.stem,
                marks=pytest.mark.sync_exec,
            ))
    return params


SYNC_SCRIPT_PARAMS = _discover_sync_scripts()

if not SYNC_SCRIPT_PARAMS:
    pytest.skip(
        "No models found in assets/models/ — run setup_sample_models.sh first",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.save_mode
@pytest.mark.parametrize("script,model,image", SYNC_SCRIPT_PARAMS)
def test_save_output(script: Path, model: Path, image: Path, tmp_path: Path):
    """Running with --save should write at least one output file."""
    if not image.exists():
        pytest.skip(f"Test image not found: {image}")

    save_dir = tmp_path / "save_output"
    save_dir.mkdir()

    cmd = [
        sys.executable, str(script),
        "--model", str(model),
        "--image", str(image),
        "--no-display",
        "--save",
        "--save-dir", str(save_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=setup_environment(),
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} --save timed out")

    assert result.returncode == 0, (
        f"{script.name} --save failed (exit {result.returncode})\n"
        f"CMD    : {' '.join(cmd)}\n"
        f"stdout : {result.stdout[-2000:]}\nstderr : {result.stderr[-2000:]}"
    )

    saved_files = list(save_dir.rglob("*"))
    saved_files = [f for f in saved_files if f.is_file()]
    assert len(saved_files) > 0, (
        f"{script.name}: --save produced no output files in {save_dir}"
    )
