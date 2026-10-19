"""C++14 conformance lives in one CMake module shared by the main build and
the dx_postprocess pip project (U-39). A scratch project proves the flag
fails a structured binding with this machine's compiler; the pip project
itself needs libdxrt, so its wheel is built by hand (Task 3), not here."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "cmake" / "dxapp_cxx14.cmake"
PIP_CMAKE = ROOT / "src" / "bindings" / "python" / "dx_postprocess" / "CMakeLists.txt"

needs_cmake = pytest.mark.skipif(shutil.which("cmake") is None or shutil.which("c++") is None,
                                 reason="needs cmake and a C++ compiler")

PROBE = """cmake_minimum_required(VERSION 3.14)
project(cxx14_probe CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include("{module}")
if(FAKE_MSVC)
    set(MSVC 1)
endif()
dxapp_cxx14_conformance_flags(FLAGS)
message(STATUS "CXX14_FLAGS=[${{FLAGS}}]")
add_library(ok OBJECT ok.cpp)
target_compile_options(ok PRIVATE ${{FLAGS}})
add_library(cxx17 OBJECT EXCLUDE_FROM_ALL cxx17.cpp)
target_compile_options(cxx17 PRIVATE ${{FLAGS}})
"""


def _project(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "CMakeLists.txt").write_text(PROBE.format(module=MODULE.as_posix()))
    (src / "ok.cpp").write_text(
        "#include <utility>\nint ok() { std::pair<int, int> p(1, 2); return p.first + p.second; }\n")
    (src / "cxx17.cpp").write_text(
        "#include <utility>\nint cxx17() { std::pair<int, int> p(1, 2); auto [a, b] = p; return a + b; }\n")
    return src


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, timeout=300)


@needs_cmake
def test_a_structured_binding_fails_the_build(tmp_path):
    build = tmp_path / "build"
    configured = _run("cmake", "-S", str(_project(tmp_path)), "-B", str(build))
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "CXX14_FLAGS=[-Werror=c++17-extensions]" in configured.stdout
    ok = _run("cmake", "--build", str(build))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = _run("cmake", "--build", str(build), "--target", "cxx17")
    assert bad.returncode != 0
    assert "c++17-extensions" in bad.stdout + bad.stderr


@needs_cmake
def test_msvc_gets_no_gcc_flag(tmp_path):
    configured = _run("cmake", "-S", str(_project(tmp_path)), "-B", str(tmp_path / "build"),
                      "-DFAKE_MSVC=ON")
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "CXX14_FLAGS=[]" in configured.stdout


def _code_lines(path: Path) -> list:
    """The file's lines, stripped, with whole-line comments dropped: a
    commented-out call must not satisfy the checks below."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def test_the_main_build_uses_the_module():
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    code = _code_lines(ROOT / "CMakeLists.txt")
    assert "include(${CMAKE_SOURCE_DIR}/cmake/dxapp_cxx14.cmake)" in code
    assert "dxapp_cxx14_conformance_flags(DXAPP_CXX14_FLAGS)" in code
    assert "add_compile_options(${DXAPP_CXX14_FLAGS})" in code
    assert 'check_cxx_compiler_flag("-Werror=c++17-extensions"' not in text  # spelled once
    assert "MSVC stays C++17" in text


def test_the_pip_project_uses_the_module():
    text = PIP_CMAKE.read_text(encoding="utf-8")
    code = _code_lines(PIP_CMAKE)
    assert 'include("${PROJECT_ROOT}/cmake/dxapp_cxx14.cmake")' in code
    call = code.index("dxapp_cxx14_conformance_flags(DX_POSTPROCESS_CXX14_FLAGS)")
    assert any(line.startswith("pybind11_add_module(dx_postprocess") for line in code[:call])
    assert "target_compile_options(dx_postprocess PRIVATE ${DX_POSTPROCESS_CXX14_FLAGS})" in code
    assert "MSVC stays C++17" in text
