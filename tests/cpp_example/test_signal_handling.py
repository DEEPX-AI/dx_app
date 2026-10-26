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

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.utils import (  # noqa: E402
    binary_path,
    setup_environment,
    cpp_exe_task_map,
    cpp_variant_image_only,
    discover_cpp_model_cases,
)
from test_helpers.constants import IMAGE_ONLY_TASKS  # noqa: E402
from test_helpers.proc import blocked_opening_a_fifo, wait_until_blocked_opening_a_fifo  # noqa: E402

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
# exe_name → task category, to exclude image-only tasks (they reject -v).
_EXE_TASK_MAP = cpp_exe_task_map(suffixes=("_sync",))


def discover_fast_sync() -> List[tuple]:
    """Discover one fast sync executable capable of video/stream input."""
    skip = ["face", "tta", "w6"]
    # A plain detector when one is present (like test_multi_loop), otherwise
    # the first stream-capable executable.
    priority = ["yolov5-s_640x640_sync", "yolov8-n_640x640_sync", "yolov5-n_640x640_sync"]
    candidates = []
    for exe_name, model_path in discover_cpp_model_cases("_sync", BIN_DIR):
        if any(s in exe_name for s in skip):
            continue
        # SIGINT is exercised against a running VIDEO inference. Image-only
        # tasks (3d_object_detection, embedding, …) reject -v and exit
        # immediately, giving zero real coverage — skip them so a
        # stream-capable model is chosen.
        if _EXE_TASK_MAP.get(exe_name) in IMAGE_ONLY_TASKS or cpp_variant_image_only(exe_name):
            continue
        candidates.append((exe_name, model_path))
    for p in priority:
        for exe_name, model_path in candidates:
            if exe_name == p:
                return [(exe_name, model_path)]
    return candidates[:1]


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
        exe_path = binary_path(BIN_DIR, executable)
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
        exe_path = binary_path(BIN_DIR, executable)
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


# ======================================================================
# One Ctrl-C under a wrapper (`timeout`) is still ONE graceful request
# ======================================================================
WRAPPED_MODEL = MODELS_DIR / "yolov5-n_640x640.dxnn"


@pytest.mark.signal_handling
@pytest.mark.parametrize("runner", ["yolov5-n_640x640_sync", "yolov5-n_640x640_async"])
def test_one_ctrl_c_under_timeout_is_graceful(runner):
    """`timeout 20 ./bin/<runner>`, then SIGINT to the whole process group -
    what one terminal Ctrl-C does. timeout forwards it to the runner again
    (direct and to the group), so the runner receives it 2-3 times within
    microseconds. That must still be one graceful request: exit 0 with the
    interrupt message and the performance summary - not rc 130.

    Default input (an image), looped so the runner is still inferring when
    the Ctrl-C arrives. (8d0b748's runners take the model from -m only, and
    hold no image window without a display, so the image alone would end
    before the signal.)"""
    import shutil
    import threading
    exe = binary_path(BIN_DIR, runner)
    if not exe.exists():
        pytest.skip("Binary not found: {}".format(runner))
    if not WRAPPED_MODEL.exists():
        pytest.skip("{} absent - run ./setup.sh --models yolov5-n_640x640".format(WRAPPED_MODEL))
    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        pytest.skip("coreutils timeout not installed")

    env = setup_environment()
    env.pop("DISPLAY", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.Popen(
        [timeout_bin, "20", str(exe), "-m", str(WRAPPED_MODEL), "-l", "100000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
        cwd=str(PROJECT_ROOT), text=True,
        start_new_session=True,  # its own process group, like a shell job
    )
    lines = []
    started = threading.Event()

    def pump():
        for line in proc.stdout:
            lines.append(line)
            if "Starting" in line:
                started.set()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        assert started.wait(60), "runner never started:\n" + "".join(lines)
        time.sleep(1.0)  # inferring the looped image
        assert proc.poll() is None, "exited before the Ctrl-C:\n" + "".join(lines)
        os.killpg(proc.pid, signal.SIGINT)  # the terminal's Ctrl-C
        rc = proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        reader.join(5)
    output = "".join(lines)
    # timeout did not expire, so it returns the runner's own status.
    assert rc == 0, "rc={} after one Ctrl-C:\n{}".format(rc, output[-1500:])
    assert "Interrupted by user" in output, output[-1500:]
    assert "PERFORMANCE SUMMARY" in output, output[-1500:]


@pytest.mark.signal_handling
@pytest.mark.parametrize("runner", ["yolov5-n_640x640_sync", "yolov5-n_640x640_async"])
def test_a_later_ctrl_c_ends_a_run_stuck_in_its_output(runner, tmp_path):
    """U-36: the runner saves its image into a FIFO nobody reads, so it hangs
    in open() - a wind-down that cannot finish. The first SIGINT is a
    request and the run stays stuck; a second one more than 200 ms later
    must end the process by SIGINT. Before the escalation only SIGKILL could."""
    import threading
    exe = binary_path(BIN_DIR, runner)
    if not exe.exists():
        pytest.skip("Binary not found: {}".format(runner))
    if not WRAPPED_MODEL.exists():
        pytest.skip("{} absent - run ./setup.sh --models yolov5-n_640x640".format(WRAPPED_MODEL))
    fifo = tmp_path / "stuck.png"
    os.mkfifo(str(fifo))
    env = setup_environment()
    env.pop("DISPLAY", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DXAPP_SAVE_IMAGE"] = str(fifo)
    proc = subprocess.Popen([str(exe), "-m", str(WRAPPED_MODEL), "--no-display"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            cwd=str(PROJECT_ROOT), text=True)
    lines = []
    started = threading.Event()

    def pump():
        for line in proc.stdout:
            lines.append(line)
            if "Starting" in line:
                started.set()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        assert started.wait(60), "runner never started:\n" + "".join(lines)
        # One image inferred; now blocked opening the FIFO.
        wait_until_blocked_opening_a_fifo(proc, output=lambda: "".join(lines))
        assert proc.poll() is None, "finished before the first Ctrl-C:\n" + "".join(lines)
        proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        assert proc.poll() is None, "one Ctrl-C ended a run stuck in open():\n" + "".join(lines)
        proc.send_signal(signal.SIGINT)
        rc = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.join(5)
    assert rc == -signal.SIGINT, "rc={}:\n{}".format(rc, "".join(lines)[-1500:])


def test_blocked_opening_a_fifo_sees_only_an_open_waiting_for_its_reader(tmp_path):
    """The FIFO tests signal once the run sleeps in open(); no NPU needed."""
    fifo = tmp_path / "wait.fifo"
    os.mkfifo(str(fifo))
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    writer = subprocess.Popen([sys.executable, "-c", "open({!r}, 'w')".format(str(fifo))])
    try:
        wait_until_blocked_opening_a_fifo(writer, timeout=10)
        assert not blocked_opening_a_fifo(sleeper.pid)
        with open(str(fifo)):  # the reader arrives: the open completes
            assert writer.wait(timeout=10) == 0
        with pytest.raises(AssertionError, match="before blocking on the FIFO"):
            wait_until_blocked_opening_a_fifo(writer, timeout=10)
    finally:
        for proc in (sleeper, writer):
            if proc.poll() is None:
                proc.kill()
            proc.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
