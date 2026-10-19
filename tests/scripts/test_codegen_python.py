"""The configure step's registry codegen needs Python 3; without one it says
so plainly (SP6 U-37). CMake reflows FATAL_ERROR text, so the assertions
compare with whitespace collapsed."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "cmake" / "dxapp_codegen_python.cmake"
CPP_CMAKE = ROOT / "src" / "cpp_example" / "CMakeLists.txt"

needs_cmake = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")

PROBE = """cmake_minimum_required(VERSION 3.14)
project(codegen_python_probe NONE)
include("{module}")
dxapp_find_codegen_python()
message(STATUS "CODEGEN_PYTHON=${{Python3_EXECUTABLE}}")
"""


def _configure(tmp_path, *extra, env=None):
    src = tmp_path / "src"
    src.mkdir()
    (src / "CMakeLists.txt").write_text(PROBE.format(module=MODULE.as_posix()))
    r = subprocess.run(["cmake", "-S", str(src), "-B", str(tmp_path / "build"), *extra],
                       capture_output=True, text=True, timeout=300, env=env)
    return r, " ".join((r.stdout + r.stderr).split())


@needs_cmake
def test_an_unusable_python_is_a_clear_configure_error(tmp_path):
    r, flat = _configure(tmp_path, "-DPython3_EXECUTABLE=/nonexistent/python3")
    assert r.returncode != 0
    assert ("No Python 3 interpreter for the graph model registry "
            "(-DPython3_EXECUTABLE=/nonexistent/python3 is not a usable Python 3).") in flat
    assert "The configure step runs scripts/gen_model_registry.py to generate it." in flat
    assert "-DPython3_EXECUTABLE=<path to python3>" in flat


@needs_cmake
def test_no_python_at_all_names_the_fix(tmp_path):
    r, flat = _configure(tmp_path, "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON")
    assert r.returncode != 0
    assert "No Python 3 interpreter for the graph model registry." in flat
    assert "https://www.python.org/downloads/" in flat


@needs_cmake
def test_a_python_is_found_and_passed_on(tmp_path):
    r, _ = _configure(tmp_path, "-DPython3_EXECUTABLE=" + sys.executable)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CODEGEN_PYTHON=" + sys.executable in r.stdout


@needs_cmake
def test_a_python_found_on_its_own_reaches_the_caller(tmp_path):
    # FindPython3 sets Python3_EXECUTABLE as a normal variable, not a cache
    # entry: only a macro hands it to the codegen's execute_process.
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    r, _ = _configure(tmp_path, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert re.search(r"CODEGEN_PYTHON=\S", r.stdout), r.stdout


def _code_lines(path):
    """The file's lines, stripped, with whole-line comments dropped: a
    commented-out call must not satisfy the checks below."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def test_the_codegen_uses_the_check():
    text = CPP_CMAKE.read_text(encoding="utf-8")
    assert "find_package(Python3 COMPONENTS Interpreter REQUIRED)" not in text
    assert "include(${PROJECT_ROOT}/cmake/dxapp_codegen_python.cmake)" in text
    assert text.index("dxapp_find_codegen_python()") < text.index(
        "COMMAND ${Python3_EXECUTABLE} ${PROJECT_ROOT}/scripts/gen_model_registry.py")
    code = _code_lines(CPP_CMAKE)
    assert "include(${PROJECT_ROOT}/cmake/dxapp_codegen_python.cmake)" in code
    assert code.index("dxapp_find_codegen_python()") < code.index(
        "COMMAND ${Python3_EXECUTABLE} ${PROJECT_ROOT}/scripts/gen_model_registry.py")
