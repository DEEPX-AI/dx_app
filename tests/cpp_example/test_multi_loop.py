"""
Test multi-loop (-l N) video functionality for C++ executables

Verifies:
  - Video multi-loop with -l 2 processes frames twice (reopen mechanism)
  - Total Frames output is approximately 2x single loop count
  - Performance summary includes correct total frame count
  - Loop banner ("Loop 1/N", "Loop 2/N") appears in output
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.utils import setup_environment  # noqa: E402

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

TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"


# ======================================================================
# Discovery
# ======================================================================
def _normalize_model_to_exe(stem: str) -> str:
    return stem.lower().replace(".", "_")


def discover_fast_sync_cases() -> List[tuple]:
    """Discover fast sync executables (no face/tta/w6)."""
    cases = []
    seen = set()
    skip_patterns = ["face", "tta", "w6"]
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        exe_name = f"{prefix}_sync"
        if any(s in exe_name for s in skip_patterns):
            continue
        if exe_name in seen:
            continue
        if (BIN_DIR / exe_name).exists():
            cases.append((exe_name, model_path))
            seen.add(exe_name)
    return sorted(cases, key=lambda x: x[0])


def _pick_one(cases: list) -> list:
    """Pick exactly one fast model for loop test."""
    priority = ["yolov5s_sync", "yolov8n_sync"]
    for exe, mp in cases:
        for p in priority:
            if exe == p:
                return [(exe, mp)]
    return cases[:1] if cases else []


FAST_SYNC = discover_fast_sync_cases()
LOOP_CASES = _pick_one(FAST_SYNC)
LOOP_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in LOOP_CASES
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.multi_loop
class TestMultiLoop:
    """Test multi-loop video processing (-l N)."""

    @pytest.mark.parametrize("executable,model_path", LOOP_PARAMS)
    def test_video_loop_2_doubles_frames(self, executable, model_path):
        """Run with -l 2 on video and verify Total Frames ≈ 2x."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        env = setup_environment()

        # Run single loop first to get baseline
        cmd_1 = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "-l", "1",
        ]
        result_1 = subprocess.run(
            cmd_1, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(PROJECT_ROOT),
        )
        assert result_1.returncode == 0, (
            f"{executable} -l 1 failed (rc={result_1.returncode})"
        )

        frames_1 = _parse_total_frames(result_1.stdout + result_1.stderr)

        # Run double loop
        cmd_2 = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "-l", "2",
        ]
        result_2 = subprocess.run(
            cmd_2, capture_output=True, text=True, timeout=1200,
            env=env, cwd=str(PROJECT_ROOT),
        )
        assert result_2.returncode == 0, (
            f"{executable} -l 2 failed (rc={result_2.returncode})"
        )

        frames_2 = _parse_total_frames(result_2.stdout + result_2.stderr)
        output_2 = result_2.stdout + result_2.stderr

        # Verify loop banners
        assert "Loop 1/" in output_2 or "loop 1/" in output_2.lower(), (
            f"No loop banner found in -l 2 output:\n{output_2[:500]}"
        )
        assert "Loop 2/" in output_2 or "loop 2/" in output_2.lower(), (
            f"No loop 2 banner found in -l 2 output:\n{output_2[:500]}"
        )

        # Verify total frames is approximately 2x baseline
        if frames_1 is not None and frames_2 is not None:
            ratio = frames_2 / max(frames_1, 1)
            assert 1.5 <= ratio <= 2.5, (
                f"Expected ~2x frames: loop1={frames_1}, loop2={frames_2}, ratio={ratio:.2f}"
            )
            print(f"\n  {executable}: loop1={frames_1} frames, loop2={frames_2} frames, ratio={ratio:.2f}x")

    @pytest.mark.parametrize("executable,model_path", LOOP_PARAMS)
    def test_image_loop_count(self, executable, model_path):
        """Run with -l N on image and verify correct iteration count."""
        loop_count = 5
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")

        test_image = SAMPLE_DIR / "img" / "sample_kitchen.jpg"
        if not test_image.exists():
            pytest.skip(f"Test image not found: {test_image}")

        env = setup_environment()
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-i", str(test_image),
            "--no-display",
            "-l", str(loop_count),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0

        output = result.stdout + result.stderr
        total_frames = _parse_total_frames(output)
        if total_frames is not None:
            assert total_frames == loop_count, (
                f"Expected {loop_count} frames, got {total_frames}"
            )
            print(f"\n  {executable}: image -l {loop_count} → {total_frames} frames ✓")

    def test_multi_loop_prerequisites(self):
        """Sanity check."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")
        assert len(LOOP_CASES) > 0, "No executables for multi-loop tests"
        print(f"\n  Loop test model: {LOOP_CASES[0][0] if LOOP_CASES else 'none'}")


# ======================================================================
# Helpers
# ======================================================================
def _parse_total_frames(output: str):
    """Parse 'Total Frames : N' from performance summary."""
    match = re.search(r"Total\s+Frames\s*:?\s*(\d+)", output, re.IGNORECASE)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
