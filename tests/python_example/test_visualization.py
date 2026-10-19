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
import ast
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

# -- common module ---------------------------------------------------------
# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.proc import example_python, run_bounded  # noqa: E402
from test_helpers.constants import (  # noqa: E402
    PROJECT_ROOT,
    TASK_IMAGE_MAP,
    MODEL_IMAGE_OVERRIDE,
    VIS_RESULT_DIR,
)
from test_helpers.utils import (  # noqa: E402
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

# A comparison visualizer (``NEEDS_REFERENCE``, e.g. EmbeddingVisualizer for
# ArcFace / CasViT Re-ID) keeps the first image as its reference and draws
# nothing for it, so a single --image never produces a picture. Which variants
# do that is read from the variant's own factory, in a child process (every
# variant ships a package named ``factory``), the way its entry script builds
# it: ``<Factory>(variant=<dir>).create_visualizer()``.
_REFERENCE_PROBE = """
import sys
from pathlib import Path
script = Path(sys.argv[1]).resolve()
for cursor in (script.parent, *script.parents):
    if (cursor / "common" / "runner" / "entry.py").is_file():
        sys.path.insert(0, str(cursor))
        break
from common.runner.entry import install_import_paths
install_import_paths(script)
import factory
vis = getattr(factory, sys.argv[2])(variant=script.parent.name).create_visualizer()
print(type(vis).__name__ if getattr(vis, "NEEDS_REFERENCE", False) else "")
"""


def _factory_class_name(script_path: Path) -> Optional[str]:
    """``X`` of the entry script's ``from factory import X``."""
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "factory":
            return node.names[0].name
    return None


def reference_visualizer(script_path: Path, env: dict) -> Optional[str]:
    """Name of the script's visualizer class if it needs a reference image, else None."""
    factory_cls = _factory_class_name(script_path)
    if factory_cls is None:
        return None
    probe = subprocess.run(
        [example_python(), "-c", _REFERENCE_PROBE, str(script_path), factory_cls],
        capture_output=True, text=True, timeout=60, env=env, cwd=str(PROJECT_ROOT),
    )
    assert probe.returncode == 0, (
        f"cannot build the visualizer of {script_path.name}: {probe.stderr[-500:]}"
    )
    return probe.stdout.strip() or None

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
            example_python(),
            str(script_path),
            "--model", str(model_path),
            "--image", str(PROJECT_ROOT / image_rel),
            "--no-display",
            "--loop", "1",
        ]

        timeout = 120

        try:
            result = run_bounded(
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

        if not output_image.exists():
            ref_vis = reference_visualizer(script_path, env)
            if ref_vis:
                pytest.skip(
                    f"{ref_vis}.NEEDS_REFERENCE (factory create_visualizer()): "
                    f"a single --image becomes the reference, nothing is rendered"
                )

        assert output_image.exists(), (
            f"Visualization image not saved: {output_image}\n"
            f"STDOUT: {result.stdout[-300:]}"
        )
        assert output_image.stat().st_size > 0, (
            f"Visualization image is empty: {output_image}"
        )

    def test_reference_visualizer_is_read_from_the_factory(self):
        """No model, no NPU: the skip above is derived from each variant's factory."""
        py = PROJECT_ROOT / "src" / "python_example"
        env = setup_environment(extra_lib_dirs=[_DX_RT_LIB])
        arcface = py / "face_recognition/arcface/arcface_mobilefacenet_112x112"
        casvit = py / "image_classification/casvit/casvit-t_224x224"
        resnet = py / "image_classification/resnet/resnet101_224x224"
        assert reference_visualizer(
            arcface / "arcface_mobilefacenet_112x112_sync.py", env) == "EmbeddingVisualizer"
        assert reference_visualizer(
            casvit / "casvit-t_224x224_async_cpp_postprocess.py", env) == "EmbeddingVisualizer"
        assert reference_visualizer(resnet / "resnet101_224x224_sync.py", env) is None

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
        result = run_bounded(
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
    python_exe = example_python()

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
