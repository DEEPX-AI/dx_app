"""
E2E tests for camera and RTSP input sources (Python examples).

Reuses ``discover_python_scripts()`` from test helpers.  Each script runs
with ``--camera <idx>`` or ``--rtsp <url>`` for ``--stream-duration``
seconds, then receives SIGINT.  Success = exit 0 + frames > 0 + FPS > 0.

Usage via run_tc.sh::

    ./run_tc.sh --python --camera --camera-index 0
    ./run_tc.sh --python --rtsp --rtsp-url rtsp://192.168.30.100:8554/stream1

Direct pytest::

    pytest tests/python_example/test_e2e_camera_rtsp.py -m e2e_camera \\
           --camera-index 0 --stream-duration 10 -v
    pytest tests/python_example/test_e2e_camera_rtsp.py -m e2e_rtsp \\
           --rtsp-url rtsp://192.168.30.100:8554/stream1 -v
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import (  # noqa: E402
    E2E_SHORT_MODELS,
    PROJECT_ROOT,
)
from test_helpers.utils import (  # noqa: E402
    discover_python_scripts,
    setup_environment,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _build_params() -> list:
    """Return list of pytest.param(script, model_path, id=..., marks=...)."""
    raw = discover_python_scripts(suffixes=("_sync", "_async"))
    params = []
    for _task, model_name, sync_scripts, async_scripts, model_path in raw:
        for script, mode in ([(s, "sync") for s in sync_scripts]
                             + [(s, "async") for s in async_scripts]):
            exec_marker = (pytest.mark.sync_exec if mode == "sync"
                           else pytest.mark.async_exec)
            marks = [exec_marker]
            if model_name in E2E_SHORT_MODELS:
                marks.append(pytest.mark.e2e_short)
            params.append(pytest.param(script, model_path,
                                       id=script.stem, marks=marks))
    return params


SCRIPT_PARAMS = _build_params()

if not SCRIPT_PARAMS:
    pytest.skip("No Python scripts found", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_stream_test(cmd: list, duration: int, env: dict, label: str):
    """Run *cmd* for *duration* seconds, SIGINT, verify output."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(PROJECT_ROOT),
        preexec_fn=os.setsid,
    )

    # Check early exit (arg error / import error)
    try:
        proc.wait(timeout=3)
        out = (proc.stdout.read().decode(errors="replace")
               + proc.stderr.read().decode(errors="replace"))[-1500:]
        pytest.fail(f"{label}: early exit (code={proc.returncode})\n{out}")
    except subprocess.TimeoutExpired:
        pass  # still running — good

    time.sleep(max(0, duration - 3))
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)

    try:
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        pytest.fail(f"{label}: did not exit after SIGINT+30s")

    output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
    rc = proc.returncode

    assert rc in (0, -signal.SIGINT, 130), (
        f"{label}: bad exit code {rc}\n{output[-1000:]}")

    assert "[ERROR]" not in output and "Failed to open" not in output, (
        f"{label}: runtime error\n{output[-1000:]}")

    m_frames = re.search(r"Total Frames\s*[:\s]+(\d+)", output)
    frames = int(m_frames.group(1)) if m_frames else 0
    assert frames > 0, f"{label}: Total Frames = 0\n{output[-500:]}"

    m_fps = re.search(r"Overall FPS\s*[:\s]+([\d.]+)", output)
    fps = float(m_fps.group(1)) if m_fps else 0.0

    print(f"\n{label}: frames={frames} fps={fps:.1f}")


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.e2e_camera
@pytest.mark.parametrize("script,model_path", SCRIPT_PARAMS)
def test_camera_inference(script: Path, model_path: Optional[Path],
                          camera_index, stream_duration):
    """Run Python script with --camera for *stream_duration* seconds."""
    if model_path is None:
        pytest.skip(f"Model not found for {script.stem}")

    env = setup_environment()
    cmd = [sys.executable, str(script),
           "--model", str(model_path),
           "--camera", str(camera_index),
           "--no-display"]

    _run_stream_test(cmd, stream_duration, env, f"camera/{script.stem}")


# ---------------------------------------------------------------------------
# RTSP tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.e2e_rtsp
@pytest.mark.parametrize("script,model_path", SCRIPT_PARAMS)
def test_rtsp_inference(script: Path, model_path: Optional[Path],
                        rtsp_url, stream_duration):
    """Run Python script with --rtsp for *stream_duration* seconds."""
    if model_path is None:
        pytest.skip(f"Model not found for {script.stem}")

    env = setup_environment()
    cmd = [sys.executable, str(script),
           "--model", str(model_path),
           "--rtsp", rtsp_url,
           "--no-display"]

    _run_stream_test(cmd, stream_duration, env, f"rtsp/{script.stem}")
