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
    MODELS_DIR,
    PROJECT_ROOT,
    SAMPLE_DIR,
)
from test_helpers.utils import discover_python_scripts, setup_environment  # noqa: E402

TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"
TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"


# ======================================================================
# Discovery — pick a few representative sync scripts
# ======================================================================

def _pick_representative(max_count: int = 3) -> List[pytest.param]:
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

    return [
        pytest.param(s, m, id=s.stem, marks=pytest.mark.sync_exec)
        for s, m in selected
    ]


DUMP_PARAMS = _pick_representative()


# ======================================================================
# Tests
# ======================================================================

@pytest.mark.dump_tensors
class TestDumpTensors:
    """Test ``--dump-tensors`` tensor debugging feature for Python scripts."""

    @pytest.mark.parametrize("script,model_path", DUMP_PARAMS)
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

    @pytest.mark.parametrize("script,model_path", DUMP_PARAMS)
    def test_dump_tensors_video(self, script: Path, model_path: Path, tmp_path: Path):
        """Run with --dump-tensors on video, verify per-frame .npy files."""
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

    def test_dump_tensors_prerequisites(self):
        """Sanity check."""
        assert len(DUMP_PARAMS) > 0, "No scripts for dump-tensors tests"
        print(f"\n  Representative sync scripts: {len(DUMP_PARAMS)}")
        for p in DUMP_PARAMS:
            print(f"    - {p.values[0].stem}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
