"""The graph targets' strictness, one function per compiler family (SP1 U-13,
SP6 U-37). Scratch projects prove what dxapp_graph_strict_flags() emits -
for MSVC too, by setting MSVC on a GCC build: no MSVC here, so the MSVC
spellings are checked for being emitted, never compiled. The text checks
keep every graph target on the function."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "cmake" / "dxapp_graph_flags.cmake"
CPP_CMAKE = ROOT / "src" / "cpp_example" / "CMakeLists.txt"
PY_CMAKE = ROOT / "src" / "bindings" / "python" / "dx_graph" / "CMakeLists.txt"

# target -> SWITCH (its code switches over enums with no default)
GRAPH_TARGETS = {
    "dxapp_graph_obj": True,
    "graph_engine_test": True,
    "dxapp_graph_models_obj": False,
    "dxapp_graph_cli_obj": False,
    "dxapp_graph_consumer_obj": False,
}

needs_cmake = pytest.mark.skipif(shutil.which("cmake") is None or shutil.which("c++") is None,
                                 reason="needs cmake and a C++ compiler")

PROBE = """cmake_minimum_required(VERSION 3.14)
project(graph_flags_probe CXX)
include("{module}")
if(FAKE_MSVC)
    set(MSVC 1)
endif()
add_library(plain OBJECT x.cpp)
add_library(with_switch OBJECT x.cpp)
if(TYPO)
    dxapp_graph_strict_flags(plain SWICH)
endif()
dxapp_graph_strict_flags(plain)
dxapp_graph_strict_flags(with_switch SWITCH)
get_target_property(plain_flags plain COMPILE_OPTIONS)
get_target_property(switch_flags with_switch COMPILE_OPTIONS)
message(STATUS "PLAIN=[${{plain_flags}}]")
message(STATUS "SWITCH=[${{switch_flags}}]")
"""


def _configure(tmp_path, *extra):
    src = tmp_path / "src"
    src.mkdir()
    (src / "CMakeLists.txt").write_text(PROBE.format(module=MODULE.as_posix()))
    (src / "x.cpp").write_text("int x() { return 0; }\n")
    return subprocess.run(["cmake", "-S", str(src), "-B", str(tmp_path / "build"), *extra],
                          capture_output=True, text=True, timeout=300)


@needs_cmake
def test_strict_flags_on_gcc_and_clang(tmp_path):
    r = _configure(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PLAIN=[-Werror=reorder]" in r.stdout
    assert "SWITCH=[-Werror=switch;-Werror=reorder]" in r.stdout


@needs_cmake
def test_strict_flags_on_msvc(tmp_path):
    r = _configure(tmp_path, "-DFAKE_MSVC=ON")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PLAIN=[/bigobj;/w15038;/we5038]" in r.stdout
    assert "SWITCH=[/bigobj;/w15038;/we5038;/w14062;/we4062]" in r.stdout


@needs_cmake
def test_a_misspelled_keyword_fails_the_configure(tmp_path):
    r = _configure(tmp_path, "-DTYPO=ON")
    assert r.returncode != 0
    flat = " ".join((r.stdout + r.stderr).split())
    assert "dxapp_graph_strict_flags(plain): unknown argument(s) SWICH" in flat


def _code_lines(path):
    """The file's lines, stripped, with whole-line comments dropped: a
    commented-out call must not satisfy the checks below."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _compile_options(text, target):
    pattern = r"target_compile_options\(\s*{}\s+PRIVATE\s+([^)]*)\)".format(re.escape(target))
    return " ".join(re.findall(pattern, text))


@pytest.mark.parametrize("target,switch", sorted(GRAPH_TARGETS.items()))
def test_graph_target_uses_the_strict_flags(target, switch):
    text = CPP_CMAKE.read_text(encoding="utf-8")
    call = "dxapp_graph_strict_flags({}{})".format(target, " SWITCH" if switch else "")
    assert call in text
    assert call in _code_lines(CPP_CMAKE)
    assert "-Werror" not in _compile_options(text, target)


def test_dx_graph_module_uses_the_strict_flags():
    text = PY_CMAKE.read_text(encoding="utf-8")
    assert "dxapp_graph_strict_flags(_dx_graph)" in text
    assert "dxapp_graph_strict_flags(_dx_graph)" in _code_lines(PY_CMAKE)
    assert "-Werror" not in _compile_options(text, "_dx_graph")


def test_no_werror_is_spelled_outside_the_module():
    for path in (CPP_CMAKE, PY_CMAKE):
        assert "-Werror=" not in path.read_text(encoding="utf-8"), path
