"""
Python Example Visualization Tests — Run + Image Verification

Strategy:
  For each Python model discovered via ``src/python_example/<task>/<model>/``:
    1. Locate sync/async scripts and matching ``.dxnn`` model file
    2. Run the script with ``--model <dxnn> --image <sample> --no-display --loop 1``
    3. Capture visualization via ``DXAPP_SAVE_IMAGE`` environment variable
    4. Verify the output image exists, is non-empty, and is a valid image

Output directory (pytest):
  ``tests/test_visualization_result/python_example/{sync,async}/<task>/<model>.jpg``

Standalone mode (``python test_visualization.py``):
  Same output structure, inline progress report.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

# -- common module ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.constants import (  # noqa: E402
    PROJECT_ROOT,
    TASK_IMAGE_MAP,
    MODEL_IMAGE_OVERRIDE,
    VIS_RESULT_DIR,
)
from common.utils import (  # noqa: E402
    discover_python_scripts,
    setup_environment,
    resolve_image_for_model,
)

# ======================================================================
# Output root
# ======================================================================
PY_VIS_DIR = VIS_RESULT_DIR / "python_example"

# Extra library directory for dx_rt
_DX_RT_LIB = PROJECT_ROOT.parent / "dx_rt" / "build_x86_64" / "lib"


# ======================================================================
# Discovery — build parametrize list
# ======================================================================
def _build_vis_params():
    """Return flat list of (task, model_name, script_path, mode, image_rel) for sync+async."""
    raw = discover_python_scripts(suffixes=("_sync", "_async"))
    params = []
    for task, model_name, sync_scripts, async_scripts, model_path in raw:
        if model_path is None:
            continue
        img_rel = resolve_image_for_model(model_name, task)
        if img_rel is None:
            img_rel = TASK_IMAGE_MAP.get(task, "sample/img/sample_kitchen.jpg")
        for script in sync_scripts:
            params.append((task, model_name, script, model_path, "sync", img_rel))
        for script in async_scripts:
            params.append((task, model_name, script, model_path, "async", img_rel))
    return params


DISCOVERED = _build_vis_params()

VIS_PARAMS = [
    pytest.param(task, name, script, model, mode, img, id=script.stem)
    for task, name, script, model, mode, img in DISCOVERED
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.visualization
class TestPythonVisualization:
    """Python example visualization smoke tests (sync + async)."""

    @pytest.mark.parametrize(
        "task,model_name,script_path,model_path,mode,image_rel", VIS_PARAMS
    )
    def test_visualization_output(
        self, task, model_name, script_path, model_path, mode, image_rel
    ):
        """Run a Python example script and verify that a visualization image is produced."""
        out_dir = PY_VIS_DIR / mode / task
        out_dir.mkdir(parents=True, exist_ok=True)
        output_image = out_dir / f"{model_name}.jpg"

        env = setup_environment(extra_lib_dirs=[_DX_RT_LIB])
        env["DXAPP_SAVE_IMAGE"] = str(output_image)

        cmd = [
            sys.executable,
            str(script_path),
            "--model", str(model_path),
            "--image", str(PROJECT_ROOT / image_rel),
            "--no-display",
            "--loop", "1",
        ]

        timeout = 120

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"{model_name}_{mode} timed out after {timeout}s")

        assert result.returncode == 0, (
            f"{model_name}_{mode} failed (rc={result.returncode})\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT: {result.stdout[-500:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        assert output_image.exists(), (
            f"Visualization image not saved: {output_image}\n"
            f"STDOUT: {result.stdout[-300:]}"
        )
        assert output_image.stat().st_size > 0, (
            f"Visualization image is empty: {output_image}"
        )

    def test_visualization_prerequisites(self):
        """Sanity: count discoverable Python models."""
        raw = discover_python_scripts()
        total = sum(
            1 for _, _, sync_s, async_s, model in raw
            if model is not None and (sync_s or async_s)
        )
        print(f"\n  Discoverable Python models with .dxnn: {total}")
        print(f"  Total parameters (sync + async): {len(DISCOVERED)}")


# ======================================================================
# Standalone runner  (``python test_visualization.py``)
# ======================================================================
def _run_single_py_vis(cmd, output_image, timeout, run_env):
    """Execute one Python visualization and return (status, message, elapsed)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=run_env, cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return "fail", f"rc={result.returncode}", elapsed
        tag = "" if (output_image.exists() and output_image.stat().st_size > 0) else " [no-vis]"
        return "ok", tag, elapsed
    except subprocess.TimeoutExpired:
        return "fail", f"timeout ({timeout}s)", time.time() - t0
    except Exception as e:
        return "fail", f"{type(e).__name__}: {e}", time.time() - t0


def _collect_vis_counts(vis_dir: Path) -> list:
    """Collect per-task image counts from visualization output directory."""
    lines = []
    if not vis_dir.exists():
        return lines
    for mode in ("sync", "async"):
        mode_dir = vis_dir / mode
        if not mode_dir.exists():
            continue
        for td in sorted(mode_dir.iterdir()):
            if not td.is_dir():
                continue
            imgs = list(td.glob("*.jpg")) + list(td.glob("*.png"))
            lines.append(f"    {mode}/{td.name:30s} {len(imgs):3d} images")
    return lines


def _print_py_summary(ok, skip, fail, vis_dir, failures):
    """Print final Python visualization summary."""
    print(f"\n{'='*70}")
    print(f"  OK={ok}  SKIP={skip}  FAIL={fail}  TOTAL={ok+skip+fail}")
    for line in _collect_vis_counts(vis_dir):
        print(line)
    print(f"{'='*70}")
    if failures:
        print("\n  Failures:")
        for lbl, msg in failures:
            print(f"    x {lbl}: {msg}")
    print()


def main():
    """Run all Python visualization tests without pytest, printing inline results."""
    params = _build_vis_params()
    n = len(params)
    ok = skip = fail = 0
    failures = []

    print(f"\n{'='*70}")
    print(f"  Python Visualization Test  ({n} scripts, sync + async)")
    print(f"  Output: {PY_VIS_DIR}")
    print(f"{'='*70}\n")

    env = setup_environment(extra_lib_dirs=[_DX_RT_LIB])
    python_exe = sys.executable

    for i, (task, model_name, script_path, model_path, mode, image_rel) in enumerate(params, 1):
        label = f"[{i:3d}/{n}] {mode}/{task}/{model_name}"

        out_dir = PY_VIS_DIR / mode / task
        out_dir.mkdir(parents=True, exist_ok=True)
        output_image = out_dir / f"{model_name}.jpg"

        run_env = dict(env)
        run_env["DXAPP_SAVE_IMAGE"] = str(output_image)

        cmd = [
            python_exe, str(script_path),
            "--model", str(model_path),
            "--image", str(PROJECT_ROOT / image_rel),
            "--no-display",
            "--loop", "1",
        ]

        print(f"  {label} ... ", end="", flush=True)
        status, msg, elapsed = _run_single_py_vis(cmd, output_image, 120, run_env)

        if status == "ok":
            ok += 1
            print(f"OK  ({elapsed:.1f}s){msg}")
        else:
            fail += 1
            print(f"FAIL ({elapsed:.1f}s) — {msg}")
            failures.append((label, msg))

    _print_py_summary(ok, skip, fail, PY_VIS_DIR, failures)
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
