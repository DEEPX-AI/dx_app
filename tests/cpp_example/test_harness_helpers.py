"""
Harness helpers (tests/test_helpers): executable discovery, the interpreter
for Python examples, headless display and bounded subprocess runs.

Discovery helper used by the verify / dump-tensors / save-mode / multi-loop /
signal-handling suites: under the family/variant layout an executable is
``<variant>_sync`` / ``<variant>_async`` and its model is the variant's
``.dxnn`` (``yolov5-s_640x640_sync`` runs ``yolov5-s_640x640.dxnn``).
"""
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers import proc, utils  # noqa: E402
from test_helpers import platform_paths  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A models dir and a bin dir of our own, not the shared assets.

    The project root moves into tmp_path too (with the real source tree linked
    in), so the model search's walk up to a ``workspace/res/models`` cannot
    reach the suite's shared model store."""
    root = tmp_path / "root"
    models = tmp_path / "models"
    bin_dir = tmp_path / "bin"
    (root / "src").mkdir(parents=True)
    (root / "src" / "cpp_example").symlink_to(PROJECT_ROOT / "src" / "cpp_example")
    models.mkdir()
    bin_dir.mkdir()
    monkeypatch.setattr(utils, "PROJECT_ROOT", root)
    monkeypatch.setattr(utils, "MODELS_DIR", models)
    monkeypatch.setattr(utils, "BIN_DIR", bin_dir)
    utils._cpp_exe_task_map_cached.cache_clear()
    yield models, bin_dir
    utils._cpp_exe_task_map_cached.cache_clear()


def _touch_exe(path: Path) -> None:
    path.write_text("")
    os.chmod(path, 0o755)


def test_variant_executable_maps_to_its_model(sandbox):
    models, bin_dir = sandbox
    assert (PROJECT_ROOT / "src" / "cpp_example" / "object_detection" / "yolov5"
            / "yolov5-s_640x640" / "yolov5-s_640x640_sync.cpp").exists()
    (models / "yolov5-s_640x640.dxnn").write_bytes(b"x")
    _touch_exe(bin_dir / "yolov5-s_640x640_sync")
    _touch_exe(bin_dir / "yolov5-s_640x640_async")

    assert utils.discover_cpp_model_cases("_sync", bin_dir) == [
        ("yolov5-s_640x640_sync", models / "yolov5-s_640x640.dxnn")]
    assert utils.discover_cpp_model_cases("_async", bin_dir) == [
        ("yolov5-s_640x640_async", models / "yolov5-s_640x640.dxnn")]


def test_missing_binary_or_model_is_not_a_case(sandbox):
    models, bin_dir = sandbox
    (models / "yolov5-s_640x640.dxnn").write_bytes(b"x")
    assert utils.discover_cpp_model_cases("_sync", bin_dir) == []  # no binary

    (models / "yolov5-s_640x640.dxnn").unlink()
    _touch_exe(bin_dir / "yolov5-s_640x640_sync")
    assert utils.discover_cpp_model_cases("_sync", bin_dir) == []  # no model


# ======================================================================
# Headless display
# ======================================================================
def test_no_display_defaults_qt_to_offscreen():
    env = {"PATH": "/bin"}
    proc.apply_headless_display(env)
    assert env["QT_QPA_PLATFORM"] == "offscreen"


@pytest.mark.parametrize("var", ["DISPLAY", "WAYLAND_DISPLAY"])
def test_a_display_leaves_qt_alone(var):
    env = {var: ":0"}
    proc.apply_headless_display(env)
    assert "QT_QPA_PLATFORM" not in env


def test_explicit_qt_platform_is_kept():
    env = {"QT_QPA_PLATFORM": "xcb"}
    proc.apply_headless_display(env)
    assert env["QT_QPA_PLATFORM"] == "xcb"


def test_setup_environment_is_headless_safe(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    assert utils.setup_environment()["QT_QPA_PLATFORM"] == "offscreen"


# ======================================================================
# Interpreter for the Python examples
# ======================================================================
def test_example_python_prefers_the_env_override(monkeypatch):
    monkeypatch.setenv(proc.TEST_PYTHON_ENV, sys.executable)
    assert proc.example_python() == sys.executable


def test_example_python_rejects_a_bad_override(monkeypatch):
    monkeypatch.setenv(proc.TEST_PYTHON_ENV, "/nonexistent/python")
    with pytest.raises(RuntimeError, match=proc.TEST_PYTHON_ENV):
        proc.example_python()


def test_example_python_defaults_to_the_dx_runtime_venv(tmp_path, monkeypatch):
    monkeypatch.delenv(proc.TEST_PYTHON_ENV, raising=False)
    venv = tmp_path / "venv-dx-runtime"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python3").symlink_to(sys.executable)
    monkeypatch.setattr(proc, "DX_RUNTIME_VENV", venv)
    # The venv path itself, not the resolved base interpreter.
    assert proc.example_python() == str(venv / "bin" / "python3")


def test_example_python_falls_back_to_the_running_interpreter(tmp_path, monkeypatch):
    monkeypatch.delenv(proc.TEST_PYTHON_ENV, raising=False)
    monkeypatch.setattr(proc, "DX_RUNTIME_VENV", tmp_path / "missing")
    assert proc.example_python() == sys.executable


# ======================================================================
# Bounded runs: SIGTERM first, SIGKILL only when ignored
# ======================================================================
def _py(code):
    return [sys.executable, "-c", code]


def test_run_bounded_returns_like_subprocess_run():
    r = proc.run_bounded(_py("import sys; print('out'); sys.exit(3)"),
                         capture_output=True, text=True, timeout=30)
    assert (r.returncode, r.stdout) == (3, "out\n")


def test_timeout_sends_sigterm_first():
    code = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: (print('term', flush=True), sys.exit(0)))\n"
            "print('ready', flush=True); time.sleep(60)")
    t0 = time.monotonic()
    with pytest.raises(proc.BoundedTimeout) as exc:
        proc.run_bounded(_py(code), capture_output=True, text=True, timeout=1, grace=20)
    assert time.monotonic() - t0 < 15
    assert isinstance(exc.value, subprocess.TimeoutExpired)
    assert not exc.value.killed
    assert exc.value.returncode == 0  # the handler's clean exit
    assert "term" in exc.value.output


def test_timeout_escalates_to_sigkill_when_sigterm_is_ignored():
    code = ("import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(60)")
    t0 = time.monotonic()
    with pytest.raises(proc.BoundedTimeout) as exc:
        proc.run_bounded(_py(code), capture_output=True, text=True, timeout=1, grace=1)
    assert time.monotonic() - t0 < 15
    assert exc.value.killed
    assert exc.value.returncode == -signal.SIGKILL


def test_timeout_also_ends_grandchildren(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = ("import subprocess, sys, time\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "open({!r}, 'w').write(str(g.pid)); time.sleep(60)").format(str(pid_file))
    with pytest.raises(proc.BoundedTimeout):
        proc.run_bounded(_py(code), capture_output=True, text=True, timeout=2, grace=5)
    gpid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(gpid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(gpid, signal.SIGKILL)
    pytest.fail("grandchild {} survived the timeout".format(gpid))


def test_only_the_w6_face_detectors_are_too_slow_for_video():
    """U-32: video tests skipped every name containing "face"; only the W6
    face detectors (3-20 s/frame, TTA ~125 s/frame on aarch64) are too slow."""
    from test_helpers.constants import video_too_slow
    assert video_too_slow("yolov7-w6-face_960x960_sync")
    assert video_too_slow("yolov7-w6-face_1280x1280_tta_async")
    assert video_too_slow("yolov7-w6-face_960x960")
    assert video_too_slow("yolov7-w6-face_960x960_sync_cpp_postprocess")
    assert video_too_slow("yolov7-w6-face_1280x1280_tta_async_cpp_postprocess")
    for name in ("scrfd-500m_640x640_sync", "yolov5-n-face_640x640_async", "yolov7-s-face_640x640_sync",
                 "retinaface_mobilenet-0.25_640x640_sync", "3ddfa-v2_mobilenetv1_120x120_sync",
                 "scrfd-500m_640x640_sync_cpp_postprocess"):
        assert not video_too_slow(name), name


def test_windows_discovery_finds_the_exe(sandbox, monkeypatch):
    models, bin_dir = sandbox
    monkeypatch.setattr(platform_paths, "current_os_name", lambda: "nt")
    (models / "yolov5-s_640x640.dxnn").write_bytes(b"x")
    _touch_exe(bin_dir / "yolov5-s_640x640_sync.exe")
    assert utils.discover_cpp_model_cases("_sync", bin_dir) == [
        ("yolov5-s_640x640_sync", models / "yolov5-s_640x640.dxnn")]


def test_windows_stream_rejecting_cases_find_the_exe(sandbox, monkeypatch):
    models, bin_dir = sandbox
    monkeypatch.setattr(platform_paths, "current_os_name", lambda: "nt")
    (models / "yolov5-s_640x640.dxnn").write_bytes(b"x")
    _touch_exe(bin_dir / "yolov5-s_640x640_sync.exe")
    assert utils.stream_rejecting_cpp_cases(frozenset({"object_detection"}), bin_dir) == [
        ("yolov5-s_640x640_sync", models / "yolov5-s_640x640.dxnn")]


def test_linux_discovery_ignores_a_windows_exe(sandbox):
    models, bin_dir = sandbox
    (models / "yolov5-s_640x640.dxnn").write_bytes(b"x")
    _touch_exe(bin_dir / "yolov5-s_640x640_sync.exe")
    assert utils.discover_cpp_model_cases("_sync", bin_dir) == []


def _help_takes_video(exe_path: Path, env: dict) -> tuple:
    """``(exe name, --help rc, whether --help lists -v/--video_path)``."""
    result = proc.run_bounded([str(exe_path), "--help"], capture_output=True, timeout=60,
                              env=env, universal_newlines=True)
    listed = re.search(r"^\s*-v, --video", result.stdout, re.MULTILINE) is not None
    return exe_path.name, result.returncode, listed


@pytest.mark.cli
@pytest.mark.help
def test_cpp_image_only_helper_agrees_with_every_binary(bin_dir):
    """Every built C++ variant example whose --help has no -v/--video_path is
    image-only to cpp_variant_image_only (its video tests skip), and the
    runner-derived reason is given only to binaries that really have no -v.
    A config.json image_only variant may still list -v (it refuses video at
    run time), so only that direction of the config reason is checked."""
    utils._cpp_exe_source_map.cache_clear()  # the real tree, not a sandbox's
    exes = {}
    for exe, source in utils._cpp_exe_source_map().items():
        path = platform_paths.binary_path(bin_dir, exe)
        if (source.parent / "config.json").is_file() and path.exists():
            exes[exe] = path
    if len(exes) < 2:
        pytest.skip(f"C++ variant examples not built in {bin_dir}")
    env = utils.setup_environment()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: _help_takes_video(p, env), exes.values()))
    bad_rc = [f"{name}: rc={rc}" for name, rc, _ in results if rc != 0]
    assert not bad_rc, "--help failed:\n" + "\n".join(bad_rc[:20])
    wrong = []
    rejecting = 0
    for name, _rc, takes_video in results:
        exe = name[:-4] if name.endswith(".exe") else name
        reason = utils.cpp_variant_image_only_reason(exe)
        if not takes_video:
            rejecting += 1
            if reason is None:
                wrong.append(f"{exe}: --help has no -v, helper says video-capable")
        elif reason is not None and reason.startswith("image-only runner"):
            wrong.append(f"{exe}: --help lists -v, helper says {reason!r}")
    assert rejecting > 0, "no built binary rejects -v; the sweep checked nothing"
    assert not wrong, f"{len(wrong)} of {len(results)} binaries:\n" + "\n".join(wrong)
