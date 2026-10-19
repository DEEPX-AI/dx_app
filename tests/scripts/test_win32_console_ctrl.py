"""The Windows branch of signal_escalation.hpp, compiled and run on Linux (U-37).

No MSVC here, so the _WIN32 code is built with g++ -D_WIN32 against
fixtures/win32_stub/windows.h - SetConsoleCtrlHandler and GetTickCount64
with their real signatures - and driven through the handler the graph CLI
installs. Proves: the branch compiles as C++14, registers one console
routine, coalesces Ctrl-C within kInterruptCoalesceMs, escalates after it,
leaves Ctrl-Break and close to the default routine, and reports a refused
registration with GetLastError(). The stub also defines the real header's
identifier macros (far, near, min, max, small), so the header must include it
with NOMINMAX and WIN32_LEAN_AND_MEAN, and run_dir.hpp must survive it.
Cannot prove: the real <windows.h>, MSVC, or the console subsystem's
threading.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_windows_console_ctrl_branch(tmp_path):
    exe = tmp_path / "win32_console_probe"
    build = subprocess.run(
        ["g++", "-std=gnu++14", "-W", "-Wall", "-Wextra", "-Werror", "-Werror=c++17-extensions",
         "-D_WIN32", "-I", str(FIXTURES / "win32_stub"), "-I", str(ROOT / "src" / "cpp_example"),
         str(FIXTURES / "win32_console_probe.cpp"), "-o", str(exe)],
        capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert " 0 failures" in run.stdout, run.stdout
    # The refused registration's warning names GetLastError(), not errno.
    assert "could not install the SIGINT handler (error 5)" in run.stderr, run.stderr
    assert "No such file" not in run.stderr, run.stderr


DXRT_INC = Path(os.environ.get("DXRT_INCLUDE_DIR", "/usr/local/include"))
OPENCV_INC = Path(os.environ.get("OPENCV_INCLUDE_DIR", "/usr/include/opencv4"))


@pytest.mark.skipif(
    shutil.which("g++") is None or not (DXRT_INC / "dxrt" / "dxrt_api.h").is_file()
    or not (OPENCV_INC / "opencv2" / "core.hpp").is_file(),
    reason="needs g++, the dxrt headers and the OpenCV headers")
def test_run_dir_header_compiles_after_the_windows_header():
    """run_dir.hpp - in every runner - includes signal_escalation.hpp and so
    <windows.h>: nothing after it may use a name the stub defines as a macro
    (far, near, min, max). -fsyntax-only; DXRT_STATIC keeps dxrt's
    __declspec(dllimport) out, which g++ on Linux does not know."""
    build = subprocess.run(
        ["g++", "-std=gnu++14", "-W", "-Wall", "-Wextra", "-Werror", "-Werror=c++17-extensions",
         "-fsyntax-only", "-D_WIN32", "-DDXRT_STATIC", "-I", str(FIXTURES / "win32_stub"),
         "-I", str(ROOT / "src" / "cpp_example"), "-isystem", str(DXRT_INC), "-isystem", str(OPENCV_INC),
         str(FIXTURES / "win32_run_dir_probe.cpp")],
        capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
