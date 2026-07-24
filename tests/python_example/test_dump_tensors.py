"""
Test --dump-tensors functionality for Python inference scripts.

Verifies:
  - ``--dump-tensors`` flag creates tensor dump directories
  - ``.npy`` files are produced for input/output tensors
  - Dump directory structure follows ``run_dir/tensors/`` pattern

Mirrors ``tests/cpp_example/test_dump_tensors.py``.
"""
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import (  # noqa: E402
    ASSETS_DIR,
    IMAGE_ONLY_TASKS,
    MODELS_DIR,
    PROJECT_ROOT,
    SAMPLE_DIR,
    STREAM_REJECTING_TASKS_PY,
)
from test_helpers.utils import discover_python_scripts, setup_environment  # noqa: E402


def _task_of(script: Path) -> str:
    """Task category for a python example script.

    Scripts live at ``src/python_example/<task>/<model>/<script>.py``, so the
    task is the grandparent directory name.
    """
    return script.parent.parent.name

TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"
TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"


# ======================================================================
# Discovery — pick a few representative sync scripts
# ======================================================================

def _pick_representative(max_count: int = 3) -> List[tuple]:
    """Representative ``(script, model)`` subset (prioritise a few fast models)."""
    raw = discover_python_scripts(suffixes=("_sync",))
    candidates = []
    for _task, model_name, sync_scripts, _async, model_path in raw:
        if model_path is None or not sync_scripts:
            continue
        candidates.append((sync_scripts[0], model_path, model_name))

    priority = ["yolov5s", "yolov8n", "fastdepth"]
    selected = []
    for script, model, name in candidates:
        for p in priority:
            if name.startswith(p) and len(selected) < max_count:
                selected.append((script, model))
                break
    for script, model, _name in candidates:
        if len(selected) >= max_count:
            break
        if (script, model) not in selected:
            selected.append((script, model))
    return selected


def _one_per_task(tasks: frozenset) -> List[tuple]:
    """One ``(script, model)`` per task in *tasks*, from all discovered scripts."""
    raw = discover_python_scripts(suffixes=("_sync",))
    seen: set = set()
    out: list = []
    for task, _model_name, sync_scripts, _async, model_path in raw:
        if model_path is None or not sync_scripts:
            continue
        if task in tasks and task not in seen:
            out.append((sync_scripts[0], model_path))
            seen.add(task)
    return out


_REPRESENTATIVE = _pick_representative()

# Image dump runs on every representative script. Video dump runs only on
# stream-capable ones: image-only tasks (SDKREQ-517) are filtered out up-front
# (proactive) so the video test never even parametrizes a script that can't take
# ``--video`` — no runtime skip, no false failure.
IMAGE_DUMP_PARAMS = [
    pytest.param(s, m, id=s.stem, marks=pytest.mark.sync_exec)
    for s, m in _REPRESENTATIVE
]
VIDEO_DUMP_PARAMS = [
    pytest.param(s, m, id=s.stem, marks=pytest.mark.sync_exec)
    for s, m in _REPRESENTATIVE
    if _task_of(s) not in IMAGE_ONLY_TASKS
]
# Negative test: one script per task whose runner HARD-REJECTS stream input, to
# assert the SDKREQ-517 exclusion is actually enforced (not merely skipped).
IMAGE_ONLY_REJECT_PARAMS = [
    pytest.param(s, m, id=s.stem, marks=pytest.mark.sync_exec)
    for s, m in _one_per_task(STREAM_REJECTING_TASKS_PY)
]


# ======================================================================
# Tests
# ======================================================================

@pytest.mark.dump_tensors
class TestDumpTensors:
    """Test ``--dump-tensors`` tensor debugging feature for Python scripts."""

    @pytest.mark.parametrize("script,model_path", IMAGE_DUMP_PARAMS)
    def test_dump_tensors_image(self, script: Path, model_path: Path, tmp_path: Path):
        """Run with --dump-tensors on image, verify .bin files produced."""
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        save_dir = tmp_path / "dump_img"
        cmd = [
            sys.executable, str(script),
            "--model", str(model_path),
            "--image", str(TEST_IMAGE),
            "--no-display",
            "--loop", "1",
            "--dump-tensors",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{script.name} --dump-tensors failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify tensors directory was created
        tensor_dirs = list(save_dir.rglob("tensors"))
        assert len(tensor_dirs) >= 1, (
            f"No tensors directory found under {save_dir}\n"
            f"Contents: {[str(p) for p in save_dir.rglob('*')]}"
        )

        # Verify .npy files were created
        npy_files = list(save_dir.rglob("*.npy"))
        assert len(npy_files) >= 1, (
            f"No .npy tensor files found under {save_dir}\n"
            f"Dump contents: {[str(p) for p in save_dir.rglob('*')]}"
        )

        # Verify npy files are non-empty
        for nf in npy_files:
            assert nf.stat().st_size > 0, f"Tensor file is empty: {nf}"

    @pytest.mark.parametrize("script,model_path", VIDEO_DUMP_PARAMS)
    def test_dump_tensors_video(self, script: Path, model_path: Path, tmp_path: Path):
        """Run with --dump-tensors on video, verify per-frame .npy files.

        Image-only tasks (SDKREQ-517) are excluded from ``VIDEO_DUMP_PARAMS``
        up-front, so this only ever runs on stream-capable scripts. That they
        reject ``--video`` is asserted separately by
        ``test_image_only_rejects_video``.
        """
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        if any(k in script.stem.lower() for k in ["face", "tta", "w6"]):
            pytest.skip(f"{script.name}: too slow for video dump in CI")

        save_dir = tmp_path / "dump_vid"
        cmd = [
            sys.executable, str(script),
            "--model", str(model_path),
            "--video", str(TEST_VIDEO),
            "--no-display",
            "--dump-tensors",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{script.name} video --dump-tensors failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        npy_files = list(save_dir.rglob("*.npy"))
        assert len(npy_files) >= 2, (
            f"Expected multiple .npy files for video frames, got {len(npy_files)}\n"
            f"Contents: {[str(p) for p in save_dir.rglob('*')][:20]}"
        )

    @pytest.mark.parametrize("script,model_path", IMAGE_ONLY_REJECT_PARAMS)
    def test_image_only_rejects_video(self, script: Path, model_path: Path):
        """SDKREQ-517: image-only single-model examples must REJECT stream input.

        Positive counterpart to the video test: instead of skipping image-only
        scripts, verify they refuse ``--video`` with a non-zero exit. Neither the
        runtime guard nor argparse needs an NPU (both run before engine init).
        """
        cmd = [
            sys.executable, str(script),
            "--model", str(model_path),
            "--video", str(TEST_VIDEO),
            "--no-display",
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            env=env, cwd=str(PROJECT_ROOT),
        )

        task = _task_of(script)
        # Two valid rejection forms, both satisfying SDKREQ-517 (video refused):
        #   (a) runtime guard — the script registers --video then refuses it with
        #       "...supports image input only..." (3d/object_pose/attribute_recog).
        #   (b) the script builds its parser with include_stream_inputs=False, so
        #       argparse rejects the unknown option: "unrecognized arguments:
        #       --video" (embedding, reid).
        # rc != 0 alone is too weak (a crash / missing model also exits non-zero),
        # so require rc != 0 AND a recognised rejection signature.
        stderr_low = result.stderr.lower()
        rejected = (
            "image input only" in stderr_low
            or "unrecognized arguments" in stderr_low
        )
        assert result.returncode != 0 and rejected, (
            f"[{task}] {script.name}: expected --video to be rejected (image-only "
            f"task, SDKREQ-517) but got rc={result.returncode}\n"
            f"STDERR: {result.stderr[-500:]}"
        )

    def test_dump_tensors_prerequisites(self):
        """Sanity check."""
        assert len(IMAGE_DUMP_PARAMS) > 0, "No scripts for dump-tensors tests"
        print(f"\n  Representative sync scripts: {len(IMAGE_DUMP_PARAMS)}")
        for p in IMAGE_DUMP_PARAMS:
            print(f"    - {p.values[0].stem}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
