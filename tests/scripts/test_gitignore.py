"""The tracked .gitignore covers the build, install, download and agent trees
(U-58). The patterns are anchored to the repository root, so a nested
include/ or lib/ holding sources stays tracked."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").returncode != 0,
    reason="needs a git work tree")

IGNORED = [
    "bin/yolov7_sync", "bin/python/dx_graph/__init__.py",
    "build/generated/graph_registry_all.cpp",
    "build_x86_64/CMakeCache.txt", "build_aarch64/release/bin/yolov7_sync",
    "lib/libdxapp_yolov5_postprocess.so", "include/dxapp/postprocess.h",
    "assets/YoloV7.dxnn", "download/models.tar.gz",
    "dx-agent-dev/20260930-000000_claude_x_y/README.md",
    ".cache/dxapp_build_selection_tests/build.bat", "htmlcov/index.html",
    "reports/report.json", ".coverage", "coverage.xml", "coverage.json",
]

NOT_IGNORED = [
    "build.sh", "build.bat", "scripts/build_target_resolver.sh",
    "cmake/toolchain.aarch64.cmake", "src/cpp_example/CMakeLists.txt",
    "src/cpp_example/common/utility/common_util.hpp",
    "src/postprocess/yolov5/yolov5_postprocess.h",
    "src/bindings/python/dx_graph/CMakeLists.txt", "tests/scripts/build_sh_sandbox.sh",
    "docs/graph_models.md",
    "src/cpp_example/some_task/some_model/include/some_model.hpp",  # nested include/
    "src/bindings/lib/helper.py",                                   # nested lib/
]


@pytest.mark.parametrize("path", IGNORED)
def test_a_generated_path_is_ignored(path):
    assert _git("check-ignore", "-q", "--no-index", path).returncode == 0, path


@pytest.mark.parametrize("path", NOT_IGNORED)
def test_a_source_path_is_not_ignored(path):
    assert _git("check-ignore", "-q", "--no-index", path).returncode == 1, path


def test_no_tracked_file_is_ignored():
    listed = _git("ls-files", "-ci", "--exclude-standard")
    assert listed.returncode == 0, listed.stderr
    assert listed.stdout.split() == [], listed.stdout
