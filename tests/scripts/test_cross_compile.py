"""Cross builds (U-38): dx_graph is skipped unless the target's Python is
given, keyed on CROSS_COMPILE - NOT CMAKE_CROSSCOMPILING, which both
toolchain files make TRUE on the native build too - and the graph engine and
CLI sources compile for aarch64 (scripts/check_cross_compile.sh)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "cmake" / "dxapp_platform.cmake"
SCRIPT = ROOT / "scripts" / "check_cross_compile.sh"
HOST = {"arm64": "aarch64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
OTHER = "aarch64" if HOST == "x86_64" else "x86_64"
CROSS_CXX = os.environ.get("DXAPP_CROSS_CXX", "aarch64-linux-gnu-g++")
DXRT_INC = Path(os.environ.get("DXRT_INCLUDE_DIR", "/usr/local/include"))
HOST_OPENCV_CONFIG = Path("/usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake")

needs_cmake = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
needs_known_host = pytest.mark.skipif(HOST not in ("x86_64", "aarch64"),
                                      reason=f"no toolchain file for host arch {HOST}")
needs_cross = pytest.mark.skipif(
    shutil.which(CROSS_CXX) is None or not (DXRT_INC / "dxrt" / "dxrt_api.h").is_file()
    or not Path("/usr/include/opencv4/opencv2/core.hpp").is_file(),
    reason=f"needs {CROSS_CXX}, the dxrt headers and the OpenCV headers")

PROBE = """cmake_minimum_required(VERSION 3.14)
project(platform_probe NONE)
include("{module}")
dxapp_detect_cross_compile()
option(DXAPP_BUILD_PYTHON_GRAPH "" ON)
option(DXAPP_CROSS_PYTHON_GRAPH "" OFF)
if(FAKE_MSVC)
  set(MSVC 1)
endif()
dxapp_python_graph_decision(BUILD WHY)
message(STATUS "CROSSCOMPILING=${{CMAKE_CROSSCOMPILING}} CROSS_COMPILE=${{CROSS_COMPILE}} BUILD=${{BUILD}}")
message(STATUS "WHY=${{WHY}}")
"""


def _decide(tmp_path, arch, *extra):
    src = tmp_path / "src"
    src.mkdir()
    (src / "CMakeLists.txt").write_text(PROBE.format(module=MODULE.as_posix()))
    r = subprocess.run(["cmake", "-S", str(src), "-B", str(tmp_path / "build"),
                        "-DCMAKE_TOOLCHAIN_FILE=" + str(ROOT / "cmake" / f"toolchain.{arch}.cmake"),
                        *extra], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return " ".join(r.stdout.split())


@needs_cmake
@needs_known_host
def test_the_native_toolchain_builds_the_module(tmp_path):
    # The trap: CMake reports cross-compiling on the native build too.
    assert "CROSSCOMPILING=TRUE CROSS_COMPILE=FALSE BUILD=ON" in _decide(tmp_path, HOST)


@needs_cmake
@needs_known_host
def test_a_cross_toolchain_skips_it_and_says_how_to_build_it(tmp_path):
    out = _decide(tmp_path, OTHER)
    assert "CROSS_COMPILE=TRUE BUILD=OFF" in out
    assert f"WHY=cross build (host {HOST}, target {OTHER})" in out
    assert "-DDXAPP_CROSS_PYTHON_GRAPH=ON" in out and "-DPython_INCLUDE_DIR=" in out


@needs_cmake
@needs_known_host
def test_the_option_alone_is_not_enough(tmp_path):
    assert "BUILD=OFF" in _decide(tmp_path, OTHER, "-DDXAPP_CROSS_PYTHON_GRAPH=ON")


@needs_cmake
@needs_known_host
def test_the_include_dir_alone_is_not_enough(tmp_path):
    # Both are needed: the option AND the target's include dir.
    assert "BUILD=OFF" in _decide(tmp_path, OTHER, "-DPython_INCLUDE_DIR=/opt/sysroot/usr/include/python3.12")


@needs_cmake
@needs_known_host
def test_msvc_skips_dx_graph(tmp_path):
    out = _decide(tmp_path, HOST, "-DFAKE_MSVC=ON")
    assert "CROSS_COMPILE=FALSE BUILD=OFF" in out
    assert "WHY=no Windows build of dx_graph (MSVC)" in out


@needs_cmake
@needs_known_host
def test_target_python_paths_build_it(tmp_path):
    out = _decide(tmp_path, OTHER, "-DDXAPP_CROSS_PYTHON_GRAPH=ON",
                  "-DPython_INCLUDE_DIR=/opt/sysroot/usr/include/python3.12")
    assert "BUILD=ON" in out


def test_the_cmake_files_use_the_module():
    top = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "dxapp_detect_cross_compile()" in top
    assert 'if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL' not in top
    cpp = (ROOT / "src" / "cpp_example" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "dxapp_python_graph_decision(DXAPP_PYTHON_GRAPH_BUILD DXAPP_PYTHON_GRAPH_WHY)" in cpp
    assert 'message(STATUS "dx_graph Python module skipped: ${DXAPP_PYTHON_GRAPH_WHY}")' in cpp
    assert "if(DXAPP_BUILD_PYTHON_GRAPH AND NOT MSVC)" not in cpp


@pytest.mark.skipif(
    HOST != "x86_64" or shutil.which("cmake") is None or shutil.which(CROSS_CXX) is None
    or not HOST_OPENCV_CONFIG.is_file() or not (DXRT_INC / "dxrt" / "dxrt_api.h").is_file()
    or not Path("/usr/local/lib/libdxrt.so").exists(),
    reason="needs an x86_64 host with aarch64-linux-gnu-g++, dxrt and the OpenCV CMake config")
def test_a_real_cross_configure_skips_dx_graph(tmp_path):
    """Configure only - nothing is compiled: the host OpenCV config stands in
    for an aarch64 sysroot (there is none here), as the host libdxrt does."""
    toolchain = tmp_path / "toolchain.cmake"
    toolchain.write_text('include("{}")\nset(OpenCV_DIR "{}")\n'.format(
        (ROOT / "cmake" / "toolchain.aarch64.cmake").as_posix(), HOST_OPENCV_CONFIG.parent.as_posix()))
    r = subprocess.run(["cmake", "-S", str(ROOT), "-B", str(tmp_path / "build"),
                        "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain), "-DCMAKE_BUILD_TYPE=release",
                        "-DDXAPP_PYTHON=" + sys.executable],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("dx_graph Python module skipped: cross build (host x86_64, target aarch64)"
            in " ".join(r.stdout.split()))


def test_a_missing_cross_compiler_is_a_missing_prerequisite():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "DXAPP_CROSS_CXX": "/nonexistent/aarch64-g++"})
    assert r.returncode == 2
    assert "not found" in r.stderr


def test_a_failing_registry_tu_is_named_relative_to_the_temp_dir(tmp_path):
    """--with-registry generates its TUs into a temp dir the EXIT trap
    deletes: a FAILED line must name the TU, not that vanished path. A fake
    compiler fails only the generated TUs, so no cross compiler is needed."""
    fake = tmp_path / "fake-cxx"
    fake.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in */generated/*.cpp) exit 1 ;; esac; done\n'
                    'exit 0\n')
    fake.chmod(0o755)
    dxrt = tmp_path / "dxrt-inc"
    (dxrt / "dxrt").mkdir(parents=True)
    (dxrt / "dxrt" / "dxrt_api.h").write_text("")
    opencv = tmp_path / "opencv-inc"
    (opencv / "opencv2").mkdir(parents=True)
    (opencv / "opencv2" / "core.hpp").write_text("")
    r = subprocess.run(["bash", str(SCRIPT), "--with-registry"], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "DXAPP_CROSS_CXX": str(fake), "DXRT_INCLUDE_DIR": str(dxrt),
                            "OPENCV_INCLUDE_DIR": str(opencv),
                            "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]})
    assert r.returncode == 1, r.stdout + r.stderr
    failed = [line for line in r.stdout.splitlines() if line.startswith("FAILED:")]
    assert failed, r.stdout + r.stderr
    for line in failed:
        assert line.startswith("FAILED: generated/") and line.endswith(".cpp does not compile for aarch64"), line


@needs_cross
def test_graph_sources_compile_for_aarch64():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dxrt headers: no x86-only code outside extern/" in r.stdout
    assert "check_cross_compile OK:" in r.stdout


@needs_cross
def test_a_cxx17_construct_fails_the_cross_check(tmp_path):
    copy = tmp_path / "tree"
    for rel in ("src/cpp_example/common", "src/cpp_example/multi_model_graph", "src/utility", "extern",
                "src/postprocess/sfa3d", "src/postprocess/superpoint", "src/postprocess/dope",
                "src/postprocess/yolopv2", "src/postprocess/vitpose"):
        shutil.copytree(ROOT / rel, copy / rel)
    target = copy / "src" / "cpp_example" / "common" / "graph" / "graph_config.cpp"
    target.write_text(target.read_text() + "\n#include <utility>\nnamespace {\n"
                      "int Sp6CrossProbe() { std::pair<int, int> p(1, 2); auto [a, b] = p; return a + b; }\n"
                      "}  // namespace\n")
    r = subprocess.run(["bash", str(SCRIPT), "--root", str(copy)], capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAILED: common/graph/graph_config.cpp does not compile for aarch64" in r.stdout
    assert "c++17-extensions" in r.stdout


def test_the_cross_check_mirrors_the_postprocess_includes():
    """The registry TUs include factories that include postprocess headers
    by bare name (vitpose_postprocessor.hpp: "vitpose_postprocess.h", SP4).
    Every root the real build adds must be on the cross check's path too."""
    import re
    cmake = (ROOT / "src" / "cpp_example" / "CMakeLists.txt").read_text(encoding="utf-8")
    block = re.search(r"set\(DXAPP_POSTPROCESS_INCLUDES\s+(.*?)\)", cmake, re.S)
    assert block is not None, "DXAPP_POSTPROCESS_INCLUDES moved"
    roots = re.findall(r"\$\{PROJECT_ROOT\}/(src/postprocess/[A-Za-z0-9_]+)", block.group(1))
    assert roots, block.group(1)
    script = SCRIPT.read_text(encoding="utf-8")
    for root in roots:
        assert '-I"$ROOT/{}"'.format(root) in script, root
