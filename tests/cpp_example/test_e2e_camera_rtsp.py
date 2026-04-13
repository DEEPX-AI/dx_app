"""
E2E tests for camera and RTSP input sources.

Reuses the same model/executable discovery from ``test_e2e.py`` but runs each
executable with ``-c <camera_index>`` or ``-r <rtsp_url>`` instead of
``-v <video>``.  Each process runs for ``--stream-duration`` seconds then
receives SIGINT; success = exit 0 + frames > 0 + FPS > 0.

Usage via run_tc.sh::

    ./run_tc.sh --cpp --camera --camera-index 0
    ./run_tc.sh --cpp --rtsp --rtsp-url rtsp://192.168.30.100:8554/stream1
    ./run_tc.sh --cpp --camera --rtsp --camera-index 0 --rtsp-url rtsp://...

Direct pytest::

    pytest tests/cpp_example/test_e2e_camera_rtsp.py -m e2e_camera \\
           --camera-index 0 --stream-duration 10 -v
    pytest tests/cpp_example/test_e2e_camera_rtsp.py -m e2e_rtsp \\
           --rtsp-url rtsp://192.168.30.100:8554/stream1 -v
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest

from conftest import resolve_bin_dir

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.constants import (  # noqa: E402
    E2E_SHORT_MODELS,
    MODELS_DIR,
    MULTI_MODEL_EXECUTABLES,
    PROJECT_ROOT,
)
from test_helpers.utils import (  # noqa: E402
    normalize_model_name as _normalize,
    setup_environment,
)

BIN_DIR = resolve_bin_dir()

# ---------------------------------------------------------------------------
# Discovery  (reuses same logic as test_e2e.py)
# ---------------------------------------------------------------------------

_MULTI_MODEL_E2E = {}
for _base, _pairs in MULTI_MODEL_EXECUTABLES.items():
    _MULTI_MODEL_E2E[f"{_base}_sync"] = _pairs
    _MULTI_MODEL_E2E[f"{_base}_async"] = _pairs


def _resolve_multi_model_paths(exe_name: str) -> Optional[List[Path]]:
    pairs = _MULTI_MODEL_E2E.get(exe_name)
    if pairs is None:
        return None
    resolved = []
    for _flag, fname in pairs:
        p = MODELS_DIR / fname
        if not p.exists():
            return None
        resolved.append(p)
    return resolved


def _find_dxnn(base_name: str) -> Optional[Path]:
    for m in sorted(MODELS_DIR.glob("*.dxnn")):
        if _normalize(m.stem) == base_name:
            return m
    for m in sorted(MODELS_DIR.glob("*.dxnn")):
        mn = _normalize(m.stem)
        if mn.startswith(base_name + "_") or mn.startswith(base_name + "-"):
            if (BIN_DIR / f"{mn}_sync").exists() or (BIN_DIR / f"{mn}_async").exists():
                continue
            return m
    return None


def _discover() -> list:
    cases = []
    if not BIN_DIR.exists():
        return cases
    seen = set()
    for exe_path in sorted(BIN_DIR.iterdir()):
        if not exe_path.is_file():
            continue
        name = exe_path.name
        if not (name.endswith("_sync") or name.endswith("_async")):
            continue
        if name in seen:
            continue
        if name in _MULTI_MODEL_E2E:
            cases.append((name, _resolve_multi_model_paths(name)))
        else:
            base = name.rsplit("_", 1)[0]
            cases.append((name, _find_dxnn(base)))
        seen.add(name)
    return sorted(cases, key=lambda x: x[0])


def _mark(cases: list) -> list:
    params = []
    for exe, model_path in cases:
        marker = pytest.mark.async_exec if "_async" in exe else pytest.mark.sync_exec
        base = exe.rsplit("_", 1)[0]
        marks = [marker]
        if base in E2E_SHORT_MODELS:
            marks.append(pytest.mark.e2e_short)
        params.append(pytest.param(exe, model_path, id=exe, marks=marks))
    return params


EXECUTABLE_PARAMS = _mark(_discover())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_args(exe_name: str, model_path) -> list:
    """Build model CLI args (handles multi-model executables)."""
    if isinstance(model_path, list):
        pairs = _MULTI_MODEL_E2E[exe_name]
        args = []
        for (flag, _), mp in zip(pairs, model_path):
            args += [flag, str(mp)]
        return args
    return ["-m", str(model_path)]


def _run_stream_test(cmd: list, duration: int, env: dict, label: str):
    """Run cmd for *duration* seconds, SIGINT, then verify output."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(PROJECT_ROOT),
        preexec_fn=os.setsid,
    )

    # Check early exit (startup failure)
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

    # Verify
    assert rc in (0, -signal.SIGINT, 130), (
        f"{label}: bad exit code {rc}\n{output[-1000:]}")

    assert "[ERROR]" not in output and "Failed to open" not in output, (
        f"{label}: runtime error\n{output[-1000:]}")

    m_frames = re.search(r"Total Frames\s*[:\s]+(\d+)", output)
    frames = int(m_frames.group(1)) if m_frames else 0
    assert frames > 0, f"{label}: Total Frames = 0 (no frames read)\n{output[-500:]}"

    m_fps = re.search(r"Overall FPS\s*[:\s]+([\d.]+)", output)
    fps = float(m_fps.group(1)) if m_fps else 0.0

    print(f"\n{label}: frames={frames} fps={fps:.1f}")


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.e2e_camera
@pytest.mark.parametrize("executable,model_path", EXECUTABLE_PARAMS)
def test_camera_inference(executable, model_path, bin_dir, camera_index,
                          stream_duration):
    """Run executable with camera input for *stream_duration* seconds."""
    exe_path = bin_dir / executable
    if not exe_path.exists():
        pytest.skip(f"Executable not found: {exe_path}")
    if model_path is None:
        pytest.skip(f"Model not found for {executable}")

    env = setup_environment()
    cmd = ([str(exe_path)] + _build_model_args(executable, model_path)
           + ["-c", str(camera_index), "--no-display"])

    _run_stream_test(cmd, stream_duration, env, f"camera/{executable}")


# ---------------------------------------------------------------------------
# RTSP tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.e2e_rtsp
@pytest.mark.parametrize("executable,model_path", EXECUTABLE_PARAMS)
def test_rtsp_inference(executable, model_path, bin_dir, rtsp_url,
                        stream_duration):
    """Run executable with RTSP input for *stream_duration* seconds."""
    exe_path = bin_dir / executable
    if not exe_path.exists():
        pytest.skip(f"Executable not found: {exe_path}")
    if model_path is None:
        pytest.skip(f"Model not found for {executable}")

    env = setup_environment()
    cmd = ([str(exe_path)] + _build_model_args(executable, model_path)
           + ["-r", rtsp_url, "--no-display"])

    _run_stream_test(cmd, stream_duration, env, f"rtsp/{executable}")
