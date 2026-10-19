"""Where the C++ binaries are on Windows - bin/Release (or another config
dir) or bin/, names ending in .exe - unit-tested on Linux (U-37)."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "platform_paths", ROOT / "tests" / "test_helpers" / "platform_paths.py")
pp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pp)


def test_linux_uses_bin_and_bare_names(tmp_path):
    (tmp_path / "bin" / "Release").mkdir(parents=True)
    assert pp.resolve_bin_dir(tmp_path, "posix") == tmp_path / "bin"
    assert pp.exe_filename("yolov7_sync", "posix") == "yolov7_sync"
    assert pp.binary_path(tmp_path / "bin", "yolov7_sync", "posix") == tmp_path / "bin" / "yolov7_sync"


def test_windows_prefers_bin_release(tmp_path):
    (tmp_path / "bin" / "Release").mkdir(parents=True)
    (tmp_path / "bin" / "Debug").mkdir()
    assert pp.resolve_bin_dir(tmp_path, "nt") == tmp_path / "bin" / "Release"


def test_windows_takes_another_config_when_release_is_absent(tmp_path):
    (tmp_path / "bin" / "Debug").mkdir(parents=True)
    assert pp.resolve_bin_dir(tmp_path, "nt") == tmp_path / "bin" / "Debug"


def test_windows_falls_back_to_bin(tmp_path):
    # build.bat's xcopy of an install tree's bin/ into bin/
    (tmp_path / "bin").mkdir()
    assert pp.resolve_bin_dir(tmp_path, "nt") == tmp_path / "bin"


def test_windows_names_end_in_exe():
    assert pp.exe_filename("yolov7_sync", "nt") == "yolov7_sync.exe"
    assert pp.exe_filename("yolov7_sync.exe", "nt") == "yolov7_sync.exe"
    assert pp.exe_filename("Graph.EXE", "nt") == "Graph.EXE"
    assert pp.binary_path(Path("b"), "x", "nt") == Path("b") / "x.exe"


def test_the_default_is_the_running_os(monkeypatch):
    monkeypatch.setattr(pp, "current_os_name", lambda: "nt")
    assert pp.exe_filename("a") == "a.exe"
    monkeypatch.setattr(pp, "current_os_name", lambda: "posix")
    assert pp.exe_filename("a") == "a"


def test_the_windows_graph_smoke_tests_skip_off_windows():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "test_graph_smoke.py", "-q", "-p", "no:cacheprovider"],
        cwd=str(ROOT / "tests" / "windows"), capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    out = completed.stdout + completed.stderr
    assert completed.returncode == 0, out
    assert re.search(r"\b\d+ skipped\b", out), out
    assert not re.search(r"\b\d+ (passed|failed|errors?)\b", out), out


def test_run_tests_bat_has_a_graph_option():
    text = (ROOT / "tests" / "windows" / "run_tests.bat").read_text(encoding="utf-8")
    assert 'if /i "%~1"=="graph" goto :run_graph' in text
    assert ":run_graph" in text and 'set "TEST_FILE=test_graph_smoke.py"' in text
