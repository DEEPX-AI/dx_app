"""
C++ Example Visualization Tests — Build + Run + Image Verification

Strategy:
  For each supported model discovered via ``src/cpp_example/<task>/*.cpp``:
    1. Locate ``bin/{model_name}_{sync,async}`` binary
    2. Run it with ``-m assets/models/{dxnn} -i {sample_image} -l 1 --no-display``
    3. Capture visualization via ``DXAPP_SAVE_IMAGE`` environment variable
    4. Verify the output image exists, is non-empty, and is a valid image

Output directory (pytest):
  ``tests/test_visualization_result/cpp_example/{sync,async}/<task>/<model>.jpg``

Standalone mode (``python test_visualization.py``):
  Same output structure, inline progress report.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

# -- common module ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.constants import (  # noqa: E402
    BIN_DIR,
    MODELS_DIR,
    MULTI_MODEL_EXECUTABLES,
    PROJECT_ROOT,
    REGISTRY_PATH,
    SAMPLE_DIR,
    VIS_RESULT_DIR,
)
from common.utils import (  # noqa: E402
    discover_cpp_executables,
    setup_environment,
)

# ======================================================================
# Output root
# ======================================================================
CPP_VIS_DIR = VIS_RESULT_DIR / "cpp_example"


# ======================================================================
# Discovery — build parametrize list
# ======================================================================
def _build_vis_params():
    """Return list of (task, exe_name, model_args, is_multi, image_rel) for all sync+async."""
    return discover_cpp_executables(suffixes=("_sync", "_async"))


DISCOVERED = _build_vis_params()

VIS_PARAMS = [
    pytest.param(task, exe, args, multi, img, id=exe)
    for task, exe, args, multi, img in DISCOVERED
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.visualization
class TestCppVisualization:
    """C++ binary visualization smoke tests (sync + async)."""

    @pytest.mark.parametrize("task,exe_name,model_args,is_multi,image_rel", VIS_PARAMS)
    def test_visualization_output(self, task, exe_name, model_args, is_multi, image_rel):
        """Run a C++ binary and verify that a visualization image is produced."""
        suffix = "async" if exe_name.endswith("_async") else "sync"
        base_name = exe_name.rsplit("_", 1)[0]

        # Persistent output path (not tmp_path — we keep results for inspection)
        out_dir = CPP_VIS_DIR / suffix / task
        out_dir.mkdir(parents=True, exist_ok=True)
        output_image = out_dir / f"{base_name}.jpg"

        cmd = [str(BIN_DIR / exe_name)] + model_args + [
            "-i", image_rel,
            "--save",
            "--no-display",
            "-l", "1",
        ]

        env = setup_environment()
        env["DXAPP_SAVE_IMAGE"] = str(output_image)

        timeout = 300 if ("tta" in base_name or "w6" in base_name) else 120

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
            pytest.fail(f"{exe_name} timed out after {timeout}s")

        assert result.returncode == 0, (
            f"{exe_name} failed (rc={result.returncode})\n"
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
        """Sanity: required directories and registry exist."""
        assert REGISTRY_PATH.exists(), f"Registry not found: {REGISTRY_PATH}"
        assert BIN_DIR.exists(), f"Bin directory not found: {BIN_DIR}"
        assert MODELS_DIR.exists(), f"Models directory not found: {MODELS_DIR}"
        assert SAMPLE_DIR.exists(), f"Sample directory not found: {SAMPLE_DIR}"

        model_files = list(MODELS_DIR.glob("*.dxnn"))
        bin_files = [f for f in BIN_DIR.iterdir() if f.is_file() and f.name.endswith("_sync")]
        print(f"\n  .dxnn files : {len(model_files)}")
        print(f"  sync binaries: {len(bin_files)}")
        print(f"  test params  : {len(DISCOVERED)}")


# ======================================================================
# Standalone runner  (``python test_visualization.py``)
# ======================================================================
def _run_single_vis(cmd, output_image, timeout, run_env, label):
    """Execute one visualization and return (status, message).

    status: 'ok', 'fail', or 'skip'.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=run_env, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            return "fail", f"rc={result.returncode}"
        if not output_image.exists() or output_image.stat().st_size == 0:
            return "fail", "no output image"
        return "ok", ""
    except subprocess.TimeoutExpired:
        return "fail", f"timeout ({timeout}s)"
    except Exception as e:
        return "fail", f"{type(e).__name__}: {e}"

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


def _print_summary(ok, skip, fail, vis_dir, failures):
    """Print the final summary table."""
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
    """Run all C++ visualization tests without pytest, printing inline results."""
    params = _build_vis_params()
    n = len(params)
    ok = skip = fail = 0
    failures = []

    print(f"\n{'='*70}")
    print(f"  C++ Visualization Test  ({n} executables, sync + async)")
    print(f"  Output: {CPP_VIS_DIR}")
    print(f"{'='*70}\n")

    env = setup_environment()

    for i, (task, exe_name, model_args, is_multi, image_rel) in enumerate(params, 1):
        suffix = "async" if exe_name.endswith("_async") else "sync"
        base_name = exe_name.rsplit("_", 1)[0]
        label = f"[{i:3d}/{n}] {suffix}/{task}/{base_name}"

        out_dir = CPP_VIS_DIR / suffix / task
        out_dir.mkdir(parents=True, exist_ok=True)
        output_image = out_dir / f"{base_name}.jpg"

        cmd = [str(BIN_DIR / exe_name)] + model_args + [
            "-i", image_rel,
            "--save",
            "--no-display",
            "-l", "1",
        ]

        run_env = dict(env)
        run_env["DXAPP_SAVE_IMAGE"] = str(output_image)
        timeout = 300 if ("tta" in base_name or "w6" in base_name) else 120

        status, msg = _run_single_vis(cmd, output_image, timeout, run_env, label)
        if status == "ok":
            ok += 1
            print(f"  OK   {label}")
        else:
            fail += 1
            print(f"  FAIL {label}  — {msg}")
            failures.append((label, msg))

    _print_summary(ok, skip, fail, CPP_VIS_DIR, failures)
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
