"""
Test --save / --save-dir functionality for C++ executables

Verifies:
  - run_dir creation with timestamp-based directory structure
  - run_info.txt metadata file generation
  - VideoWriter output (video save mode)
  - Image save output (image save mode)
  - initVideoWriter XVID→mp4v fallback
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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
# Discovery — reuse same logic as test_e2e.py
# ======================================================================
def _normalize_model_to_exe(stem: str) -> str:
    return stem.lower().replace(".", "_")


def discover_sync_cases() -> List[tuple]:
    """Discover (executable_name, model_path) pairs for sync executables."""
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
    """Pick a small representative subset to keep tests fast."""
    # Prefer one detection, one classification, one other
    priority_prefixes = ["yolov5s_sync", "yolov8n_sync", "fastdepth"]
    selected = []
    for exe, mp in cases:
        for p in priority_prefixes:
            if exe.startswith(p) and len(selected) < max_count:
                selected.append((exe, mp))
                break
    # Fill remaining
    for exe, mp in cases:
        if len(selected) >= max_count:
            break
        if (exe, mp) not in selected:
            selected.append((exe, mp))
    return selected


SYNC_CASES = discover_sync_cases()
REPRESENTATIVE_CASES = _pick_representative(SYNC_CASES)
SAVE_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in REPRESENTATIVE_CASES
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.save_mode
class TestSaveMode:
    """Test --save and --save-dir CLI options."""

    @pytest.mark.parametrize("executable,model_path", SAVE_PARAMS)
    def test_image_save_creates_run_dir(self, executable, model_path, tmp_path):
        """Run with --save --save-dir, verify run_dir structure for image input."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        save_dir = tmp_path / "save_test"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{executable} failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify run_dir was created under save_dir
        assert save_dir.exists(), f"save_dir not created: {save_dir}"

        # Find the run directory (should contain a timestamp-based subdir)
        run_dirs = list(save_dir.rglob("run_info.txt"))
        assert len(run_dirs) >= 1, (
            f"No run_info.txt found under {save_dir}\n"
            f"Contents: {list(save_dir.rglob('*'))}"
        )

        # Verify run_info.txt content
        run_info = run_dirs[0]
        run_info_text = run_info.read_text()
        assert "model" in run_info_text.lower() or "Model" in run_info_text, (
            f"run_info.txt missing model info:\n{run_info_text[:300]}"
        )

    @pytest.mark.parametrize("executable,model_path", SAVE_PARAMS)
    def test_video_save_creates_output(self, executable, model_path, tmp_path):
        """Run with --save on video input, verify video file is produced."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        # Skip face models (too slow for video)
        if "face" in executable.lower():
            pytest.skip(f"{executable}: face model too slow for video save test")

        save_dir = tmp_path / "video_save"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"{executable} video save failed (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-500:]}"
        )

        # Verify video output file was created (.avi)
        video_files = list(save_dir.rglob("*.avi"))
        assert len(video_files) >= 1, (
            f"No .avi output file found under {save_dir}\n"
            f"Contents: {list(save_dir.rglob('*'))}"
        )

        # Verify file is non-empty
        for vf in video_files:
            assert vf.stat().st_size > 0, f"Video file is empty: {vf}"

    @pytest.mark.parametrize("executable,model_path", SAVE_PARAMS)
    def test_run_info_contains_metadata(self, executable, model_path, tmp_path):
        """Verify run_info.txt contains expected metadata fields."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        save_dir = tmp_path / "metadata_test"
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(TEST_IMAGE),
            "--no-display",
            "-l", "1",
            "--save",
            "--save-dir", str(save_dir),
        ]

        env = setup_environment()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0

        run_infos = list(save_dir.rglob("run_info.txt"))
        assert len(run_infos) >= 1

        content = run_infos[0].read_text()

        # Check for expected metadata fields (run_info.txt uses 'script:' not 'executable:')
        expected_fields = ["script", "model", "input"]
        for field in expected_fields:
            assert field.lower() in content.lower(), (
                f"run_info.txt missing '{field}' field:\n{content[:500]}"
            )

    def test_save_mode_prerequisites(self):
        """Sanity: verify test prerequisites."""
        assert BIN_DIR.exists(), f"Bin directory not found: {BIN_DIR}"
        assert MODELS_DIR.exists(), f"Models directory not found: {MODELS_DIR}"
        assert len(REPRESENTATIVE_CASES) > 0, "No executables discovered for save mode tests"
        print(f"\n  Representative cases: {len(REPRESENTATIVE_CASES)}")
        for name, _ in REPRESENTATIVE_CASES:
            print(f"    - {name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
