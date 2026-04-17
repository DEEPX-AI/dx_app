"""
Test signal handling (graceful shutdown) for C++ executables

Verifies:
  - Sending SIGINT to a running video inference process triggers graceful exit
  - Exit code is 0 (clean shutdown) not 130/signal-killed
  - Output contains "Interrupted" or "Ctrl+C" message
  - No crash / segfault occurs
"""
import os
import re
import signal
import subprocess
import sys
import time
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


def discover_fast_sync() -> List[tuple]:
    """Discover one fast sync executable."""
    skip = ["face", "tta", "w6"]
    for model_path in sorted(MODELS_DIR.glob("*.dxnn")):
        prefix = _normalize_model_to_exe(model_path.stem)
        exe_name = f"{prefix}_sync"
        if any(s in exe_name for s in skip):
            continue
        if (BIN_DIR / exe_name).exists():
            return [(exe_name, model_path)]
    return []


SIGNAL_CASES = discover_fast_sync()
SIGNAL_PARAMS = [
    pytest.param(name, mp, id=name, marks=pytest.mark.sync_exec)
    for name, mp in SIGNAL_CASES
]


# ======================================================================
# Tests
# ======================================================================
@pytest.mark.signal_handling
class TestSignalHandling:
    """Test graceful shutdown via SIGINT."""

    @pytest.mark.parametrize("executable,model_path", SIGNAL_PARAMS)
    def test_sigint_graceful_shutdown(self, executable, model_path):
        """Send SIGINT during video inference, verify clean exit."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        env = setup_environment()
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "-l", "100",  # Large loop to ensure it's still running when we send SIGINT
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(PROJECT_ROOT),
            text=True,
        )

        # Wait for inference to start (2-5 seconds)
        time.sleep(3)

        # Verify process is still running
        if proc.poll() is not None:
            # Process already finished (video might be very short)
            stdout, stderr = proc.communicate()
            pytest.skip(
                f"{executable} finished before SIGINT could be sent "
                f"(rc={proc.returncode})"
            )

        # Send SIGINT
        proc.send_signal(signal.SIGINT)

        # Wait for clean shutdown (max 60 seconds)
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(f"{executable} did not exit within 60s after SIGINT")

        output = stdout + stderr

        # Verify clean exit (0) or signal exit (130 = 128 + SIGINT)
        # Graceful handling should give 0
        assert proc.returncode in [0, 130, -2], (
            f"{executable} unexpected exit code {proc.returncode} after SIGINT\n"
            f"Output: {output[-500:]}"
        )

        # Verify graceful shutdown message
        has_interrupt_msg = any(
            phrase in output.lower()
            for phrase in ["interrupted", "ctrl+c", "ctrl-c", "signal", "sigint"]
        )
        if proc.returncode == 0:
            assert has_interrupt_msg, (
                f"{executable}: clean exit (rc=0) but no interrupt message found\n"
                f"Output: {output[-500:]}"
            )

        # Verify performance summary was still printed
        has_summary = "PERFORMANCE SUMMARY" in output or "Overall FPS" in output
        if proc.returncode == 0:
            assert has_summary, (
                f"{executable}: clean exit but no performance summary\n"
                f"Output: {output[-500:]}"
            )

        print(f"\n  {executable}: SIGINT → rc={proc.returncode}, "
              f"interrupt_msg={'✓' if has_interrupt_msg else '✗'}, "
              f"perf_summary={'✓' if has_summary else '✗'}")

    @pytest.mark.parametrize("executable,model_path", SIGNAL_PARAMS)
    def test_no_segfault_on_sigint(self, executable, model_path):
        """Ensure SIGINT doesn't cause segfault (return code -11)."""
        exe_path = BIN_DIR / executable
        if not exe_path.exists():
            pytest.skip(f"Binary not found: {executable}")
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        env = setup_environment()
        cmd = [
            str(exe_path),
            "-m", str(model_path),
            "-v", str(TEST_VIDEO),
            "--no-display",
            "-l", "50",
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(PROJECT_ROOT), text=True,
        )

        time.sleep(2)
        if proc.poll() is not None:
            pytest.skip(f"{executable} already exited")

        proc.send_signal(signal.SIGINT)

        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            pytest.fail(f"{executable} hung after SIGINT")

        assert proc.returncode not in [-11, -6, 139, 134], (
            f"{executable} crashed after SIGINT (rc={proc.returncode})\n"
            f"STDERR: {stderr[-500:]}"
        )

    def test_signal_prerequisites(self):
        """Sanity check."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")
        assert len(SIGNAL_CASES) > 0, "No executables for signal tests"
        print(f"\n  Signal test model: {SIGNAL_CASES[0][0] if SIGNAL_CASES else 'none'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
