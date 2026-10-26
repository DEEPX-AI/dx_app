"""Every registry variant is a C++ and Python example the e2e suite can collect.

Discovery is the source tree plus ``bin/<variant>_{sync,async}``. A variant
missing from either tree, or from ``bin/`` once that directory has been built,
never becomes a pytest parameter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from test_helpers.utils import cpp_exe_task_map, discover_python_scripts  # noqa: E402

REGISTRY = ROOT / "config" / "model_registry.json"
BIN_DIR = ROOT / "bin"
# Model Zoo 2.5.0 additions live here until the harvested snapshot absorbs them.
SNAPSHOT = ROOT / "tests" / "data" / "modelzoo_cv_tree.json"


def _registry_variants() -> set[str]:
    rows = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {row["variant"] for row in rows if row.get("variant")}


def test_registry_variants_have_cpp_and_python_examples():
    variants = _registry_variants()
    assert len(variants) >= 497

    cpp_sync = {
        name[: -len("_sync")]
        for name in cpp_exe_task_map(("_sync",))
        if name.endswith("_sync")
    }
    python_variants = {row[1] for row in discover_python_scripts(("_sync", "_async"))}

    missing_cpp = sorted(variants - cpp_sync)
    missing_py = sorted(variants - python_variants)
    assert missing_cpp == [], f"registry variants with no C++ *_sync.cpp: {missing_cpp}"
    assert missing_py == [], f"registry variants with no Python example: {missing_py}"


def test_built_binaries_cover_every_registry_variant():
    """``bin/`` is what C++ pytest parametrizes. Skip when nothing is built."""
    if not BIN_DIR.is_dir() or not any(BIN_DIR.glob("*_sync")):
        pytest.skip("bin/ has no *_sync executables")

    missing = sorted(
        variant
        for variant in _registry_variants()
        if (
            not (BIN_DIR / f"{variant}_sync").is_file()
            or not (BIN_DIR / f"{variant}_async").is_file()
        )
    )
    assert missing == [], f"built bin/ is missing registry variants: {missing}"


def test_modelzoo_2_5_0_additions_are_in_both_example_trees():
    if not SNAPSHOT.is_file():
        pytest.skip("modelzoo snapshot is not in this checkout")
    provisional = set(
        json.loads(SNAPSHOT.read_text(encoding="utf-8"))["provisional"]["variants"]
    )
    assert len(provisional) >= 147
    variants = _registry_variants()
    assert provisional <= variants
