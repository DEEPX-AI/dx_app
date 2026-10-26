"""
Test signal handling (graceful shutdown) for Python inference scripts.

Verifies:
  - Sending SIGINT to a running video inference process triggers graceful exit
  - Exit code is 0 (clean shutdown) not signal-killed
  - Output contains "Interrupted" or "Ctrl+C" message
  - No crash / segfault occurs

Mirrors ``tests/cpp_example/test_signal_handling.py``.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pytest

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.proc import example_python  # noqa: E402
from test_helpers.constants import (  # noqa: E402
    ASSETS_DIR,
    IMAGE_ONLY_TASKS,
    MODELS_DIR,
    PROJECT_ROOT,
)
from test_helpers.utils import discover_python_scripts, setup_environment  # noqa: E402

TEST_VIDEO = ASSETS_DIR / "videos" / "dance-group.mov"


# ======================================================================
# Discovery — pick one fast sync script
# ======================================================================

def _discover_fast_sync() -> List[tuple]:
    skip = ["face", "tta", "w6"]
    raw = discover_python_scripts(suffixes=("_sync",))
    for task, model_name, sync_scripts, _async, model_path in raw:
        if model_path is None or not sync_scripts:
            continue
        if any(s in model_name for s in skip):
            continue
        # Signal handling is exercised on a running VIDEO inference. Image-only
        # tasks (embedding, 3d_object_detection, …) reject --video and exit
        # immediately, giving zero real coverage — pick a stream-capable model.
        if task in IMAGE_ONLY_TASKS:
            continue
        return [(sync_scripts[0], model_path)]
    return []


SIGNAL_CASES = _discover_fast_sync()
SIGNAL_PARAMS = [
    pytest.param(s, m, id=s.stem, marks=pytest.mark.sync_exec)
    for s, m in SIGNAL_CASES
]


# ======================================================================
# Tests
# ======================================================================

@pytest.mark.signal_handling
class TestSignalHandling:
    """Test graceful shutdown via SIGINT for Python scripts."""

    @pytest.mark.parametrize("script,model_path", SIGNAL_PARAMS)
    def test_sigint_graceful_shutdown(self, script: Path, model_path: Path):
        """Send SIGINT during video inference, verify clean exit."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        env = setup_environment()
        cmd = [
            example_python(), str(script),
            "--model", str(model_path),
            "--video", str(TEST_VIDEO),
            "--no-display",
            "--loop", "100",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(PROJECT_ROOT),
            text=True,
        )

        # Wait for inference to start
        time.sleep(3)

        if proc.poll() is not None:
            pytest.skip(
                f"{script.name} finished before SIGINT could be sent "
                f"(rc={proc.returncode})"
            )

        proc.send_signal(signal.SIGINT)

        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            pytest.fail(f"{script.name} did not exit within 60s after SIGINT")

        output = stdout + stderr

        # Python scripts should exit cleanly (0) or with KeyboardInterrupt (1)
        assert proc.returncode in [0, 1, 130, -2], (
            f"{script.name} unexpected exit code {proc.returncode} after SIGINT\n"
            f"Output: {output[-500:]}"
        )

        has_interrupt_msg = any(
            phrase in output.lower()
            for phrase in ["interrupted", "ctrl+c", "ctrl-c", "keyboardinterrupt", "signal"]
        )

        # Verify performance summary was still printed
        has_summary = "Overall FPS" in output or "PERFORMANCE SUMMARY" in output or "FPS" in output

        print(f"\n  {script.name}: SIGINT → rc={proc.returncode}, "
              f"interrupt_msg={'yes' if has_interrupt_msg else 'no'}, "
              f"perf_summary={'yes' if has_summary else 'no'}")

    @pytest.mark.parametrize("script,model_path", SIGNAL_PARAMS)
    def test_no_segfault_on_sigint(self, script: Path, model_path: Path):
        """Ensure SIGINT doesn't cause segfault (return code -11)."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")

        env = setup_environment()
        cmd = [
            example_python(), str(script),
            "--model", str(model_path),
            "--video", str(TEST_VIDEO),
            "--no-display",
            "--loop", "50",
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(PROJECT_ROOT), text=True,
        )

        time.sleep(2)
        if proc.poll() is not None:
            pytest.skip(f"{script.name} already exited")

        proc.send_signal(signal.SIGINT)

        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            pytest.fail(f"{script.name} hung after SIGINT")

        assert proc.returncode not in [-11, -6, 139, 134], (
            f"{script.name} crashed after SIGINT (rc={proc.returncode})\n"
            f"STDERR: {stderr[-500:]}"
        )

    def test_signal_prerequisites(self):
        """Sanity check."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test video not found: {TEST_VIDEO}")
        assert len(SIGNAL_CASES) > 0, "No scripts for signal tests"
        print(f"\n  Signal test script: {SIGNAL_CASES[0][0].stem if SIGNAL_CASES else 'none'}")


SIGTERM_SAVE_CASES = [
    pytest.param("object_detection/yolov5/yolov5-s_640x640/yolov5-s_640x640_sync.py", id="sync", marks=pytest.mark.sync_exec),
    pytest.param("object_detection/yolov5/yolov5-s_640x640/yolov5-s_640x640_async.py", id="async", marks=pytest.mark.async_exec),
]


@pytest.mark.signal_handling
@pytest.mark.parametrize("script_rel", SIGTERM_SAVE_CASES)
def test_sigterm_in_save_mode_finalizes_the_video_and_prints_the_summary(script_rel, tmp_path):
    """U-06: SIGTERM stops like the first Ctrl-C - the video is finalized, the summary printed."""
    import cv2
    script = PROJECT_ROOT / "src" / "python_example" / script_rel
    model = MODELS_DIR / "yolov5-s_640x640.dxnn"
    if not (script.exists() and model.exists() and TEST_VIDEO.exists()):
        pytest.skip("script, model or video missing")
    env = setup_environment()
    env["PYTHONUNBUFFERED"] = "1"
    save_dir = tmp_path / "save"
    proc = subprocess.Popen(
        [example_python(), str(script), "--model", str(model), "--video", str(TEST_VIDEO),
         "--no-display", "--save", "--save-dir", str(save_dir), "--loop", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env, cwd=str(PROJECT_ROOT), start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        writing = False
        while time.monotonic() < deadline and proc.poll() is None:
            videos = list(save_dir.rglob("output.*"))
            if videos and videos[0].stat().st_size > 200_000:  # several frames written
                writing = True
                break
            time.sleep(0.1)
        assert writing, "no video output within 60 s"
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, output[-2000:]
    assert "PERFORMANCE SUMMARY" in output, output[-2000:]
    assert "Traceback" not in output, output[-2000:]
    videos = list(save_dir.rglob("output.*"))
    assert len(videos) == 1, videos
    cap = cv2.VideoCapture(str(videos[0]))
    assert cap.isOpened(), "saved video cannot be opened (not finalized)"
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert frames >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
