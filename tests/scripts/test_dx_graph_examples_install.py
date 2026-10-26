"""D6: the dx_graph example ships next to the package, as
bin/python/dx_graph_examples/run_graph.py."""
from __future__ import annotations

import filecmp
import platform
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DX_GRAPH_SRC = ROOT / "src" / "bindings" / "python" / "dx_graph"
EXAMPLE = DX_GRAPH_SRC / "examples" / "run_graph.py"
ARCH = {"arm64": "aarch64"}.get(platform.machine(), platform.machine())
BUILD = ROOT / f"build_{ARCH}"


def test_cmake_installs_the_example_beside_the_package():
    text = (DX_GRAPH_SRC / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "DESTINATION bin/python/dx_graph_examples" in text
    assert "run_graph.py" in text


def test_readme_names_the_installed_example():
    assert "bin/python/dx_graph_examples/run_graph.py" in (DX_GRAPH_SRC / "README.md").read_text(
        encoding="utf-8")


@pytest.mark.skipif(not (BUILD / "python" / "dx_graph").is_dir(),
                    reason=f"{BUILD.name} has no dx_graph package (module not built)")
def test_build_tree_holds_the_example():
    built = BUILD / "python" / "dx_graph_examples" / "run_graph.py"
    assert built.is_file(), f"{built} missing: re-run ninja -C {BUILD.name}"
    assert filecmp.cmp(built, EXAMPLE, shallow=False), f"{built} is stale"


@pytest.mark.skipif(not (BUILD / "python" / "dx_graph").is_dir(),
                    reason=f"{BUILD.name} has no dx_graph package (module not built)")
def test_install_rules_include_the_example():
    scripts = [p for p in BUILD.rglob("cmake_install.cmake")
               if "bin/python/dx_graph" in p.read_text(errors="replace")]
    assert scripts, "no install script for the dx_graph package"
    assert any("bin/python/dx_graph_examples" in p.read_text(errors="replace")
               and "run_graph.py" in p.read_text(errors="replace") for p in scripts)
