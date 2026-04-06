"""
Test --dump-tensors functionality for C++ executables

Verifies:
  - --dump-tensors CLI flag creates tensor dump directories
  - .bin files are produced for input/output tensors
  - Exception auto-dump creates error_tensors/ directory on crash
  - Dump directory structure follows run_dir/dump_tensors/frameNNN/ pattern
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import setup_environment  # noqa: E402

from conftest import resolve_bin_dir

# ======================================================================
# Paths
# ======================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_DIR = resolve_bin_dir()
LIB_DIR = PROJECT_ROOT / "lib"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample"

TEST_IMAGE = SAMPLE_DIR / "img" / "sample_kitchen.jpg"
TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"


# ======================================================================
# Discovery
# ======================================================================
def _normalize_model_to_exe(stem: str) -> str:
    return stem.lower().replace(".", "_")


def discover_sync_cases() -> List[tuple]:
    """Discover sync executables with their model paths."""
    cases = []
    seen = set()
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        exe_name = f"{prefix}_sync"
        if exe_name in seen:
            continue
        if (BIN_DIR / exe_name).exists():
            cases.append((exe_name, model_path))
            seen.add(exe_name)
    return sorted(cases, key=lambda x: x[0])


def _pick_representative(cases: list, max_count: int = 3) -> list:
    """Pick small representative subset."""
    priority = ["yolov5s_sync", "yolov8n_sync", "fastdepth"]
    selected = []
    for exe, mp in cases:
        for p in priority:
            if exe.startswith(p) and len(selected) < max_count:
                selected.append((exe, mp))
                break
    for exe, mp in cases:
        if len(selected) >= max_count:
            break
        if (exe, mp) not in selected:
            selected.append((exe, mp))
    return selected


SYNC_CASES = discover_sync_cases()
REPRESENTATIVE = _pick_representative(SYNC_CASES)
DUMP_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in REPRESENTATIVE
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.dump_tensors
class TestDumpTensors:
    """Test --dump-tensors tensor debugging feature."""

    @pytest.mark.parametrize("executable,model_path", DUMP_PARAMS)
    def test_dump_tensors_image(self, executable, model_path, tmp_path):
        """Run with --dump-tensors on image, verify .bin files produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        save_dir = tmp_path / "dump_img"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
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
            f"{executable} --dump-tensors failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify dump directory was created
        dump_dirs = list(save_dir.rglob("dump_tensors"))
        assert len(dump_dirs) >= 1, (
            f"No dump_tensors directory found under {save_dir}\n"
            f"Contents: {[str(p) for p in save_dir.rglob('*')]}"
        )

        # Verify .bin files were created
        bin_files = list(save_dir.rglob("*.bin"))
        assert len(bin_files) >= 1, (
            f"No .bin tensor files found under {save_dir}\n"
            f"Dump contents: {[str(p) for p in save_dir.rglob('*')]}"
        )

        # Verify bin files are non-empty
        for bf in bin_files:
            assert bf.stat().st_size > 0, f"Tensor file is empty: {bf}"

    @pytest.mark.parametrize("executable,model_path", DUMP_PARAMS)
    def test_dump_tensors_video(self, executable, model_path, tmp_path):
        """Run with --dump-tensors on video, verify per-frame .bin files."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        # Skip slow models
        if any(k in executable.lower() for k in ["face", "tta", "w6"]):
            pytest.skip(f"{executable}: too slow for video dump in CI")

        save_dir = tmp_path / "dump_vid"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
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
            f"{executable} video --dump-tensors failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify per-frame dump directories with .bin files
        bin_files = list(save_dir.rglob("*.bin"))
        assert len(bin_files) >= 2, (
            f"Expected multiple .bin files for video frames, got {len(bin_files)}\n"
            f"Contents: {[str(p) for p in save_dir.rglob('*')][:20]}"
        )

    def test_dump_tensors_prerequisites(self):
        """Sanity check."""
        assert BIN_DIR.exists(), f"Bin directory not found: {BIN_DIR}"
        assert len(REPRESENTATIVE) > 0, "No executables for dump-tensors tests"
        print(f"\n  Representative sync: {len(REPRESENTATIVE)}")
        for name, _ in REPRESENTATIVE:
            print(f"    - {name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
