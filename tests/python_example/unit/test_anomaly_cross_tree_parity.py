# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The two trees must agree on what an anomaly map IS.

They did not. Python moved the efficientad family onto EfficientADPostprocessor --
teacher, student and autoencoder combined, which is the published method -- while the
C++ tree kept one network's feature magnitude behind a DepthResult and the depth
runner. Measured on the same frame and the same NPU outputs, that produced score
0.8992 against 0.0051 and a visibly different render (MAGMA at alpha 1.0, no label).

Numbers cannot be checked here -- that needs the NPU, and the cross-tree comparison is
run with DXAPP_VERIFY=1 -- so this pins the STRUCTURE that made them diverge: the
pieces exist in both trees, both anomaly families are wired to them, and neither is
routed back through the depth runner.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CPP = PROJECT_ROOT / "src" / "cpp_example"
PY = PROJECT_ROOT / "src" / "python_example"
FAMILIES = ("efficientad", "patchcore")


def _read(path: Path) -> str:
    assert path.is_file(), f"missing: {path.relative_to(PROJECT_ROOT)}"
    return path.read_text(encoding="utf-8")


def test_both_trees_carry_the_three_network_postprocessor():
    assert "0.5" in _read(CPP / "common/processors/efficientad_postprocessor.hpp")
    assert "class EfficientADPostprocessor" in _read(
        CPP / "common/processors/efficientad_postprocessor.hpp")
    assert "class EfficientADPostprocessor" in _read(
        PY / "common/processors/efficientad_postprocessor.py")


def test_both_trees_carry_an_anomaly_visualizer():
    """The depth visualizer is opaque with MAGMA, which hides the frame."""
    cpp = _read(CPP / "common/visualizers/anomaly_visualizer.hpp")
    py = _read(PY / "common/visualizers/anomaly_visualizer.py")
    for text, name in ((cpp, "C++"), (py, "Python")):
        assert "JET" in text, name
        assert "relative severity" in text, name


def _family_files(tree: Path, family: str, suffix: str) -> list[Path]:
    """Every per-variant file of *family* ending in *suffix*, in either layout.

    The tree moved from one entry per FAMILY (`efficientad_sync.cpp`) to one per
    VARIANT (`<stem>/<stem>_sync.cpp`) while this test pinned the family paths, so it
    failed on a tree that had simply been reorganised. Globbing for the suffix keeps
    the check about WHAT is wired, which is what it exists to protect.
    """
    base = tree / "anomaly_detection" / family
    found = sorted(base.rglob(f"*{suffix}"))
    assert found, f"no *{suffix} under {base.relative_to(PROJECT_ROOT)}"
    return found


def test_the_anomaly_families_do_not_use_the_depth_runner():
    for family in FAMILIES:
        for kind in ("sync", "async"):
            entries = _family_files(CPP, family, f"_{kind}.cpp")
            for entry in entries:
                text = _read(entry)
                assert f"{kind}_anomaly_runner.hpp" in text, entry
                assert "depth_runner" not in text, entry


def test_the_anomaly_factories_produce_an_anomaly_result():
    for family in FAMILIES:
        for factory in _family_files(CPP, family, "_factory.hpp"):
            text = _read(factory)
            assert "IAnomalyDetectionFactory" in text, factory
            assert "AnomalyVisualizer" in text, factory
            assert "DepthResult" not in text, factory


def test_efficientad_declares_its_companions_in_both_trees():
    """One -m, three engines: the factory is what says so."""
    for tree, suffix, needle in ((CPP, "_factory.hpp", "getCompanionModels"),
                                 (PY, "_factory.py", "get_companion_models")):
        factories = _family_files(tree, "efficientad", suffix)
        assert any(needle in _read(f) for f in factories), (
            f"no {needle} in any efficientad factory under {tree.name}")


def test_both_runners_resolve_companions_and_patchcore_declares_none():
    assert "resolveCompanionModels" in _read(
        CPP / "common/runner/sync_anomaly_runner.hpp")
    assert "resolveCompanionModels" in _read(
        CPP / "common/runner/async_anomaly_runner.hpp")
    assert "resolve_companion_models" in _read(PY / "common/runner/sync_runner.py")
    # PatchCore is one network on purpose: its metric needs a memory bank, not a model.
    for factory in _family_files(CPP, "patchcore", "_factory.hpp"):
        assert "getCompanionModels" not in _read(factory), factory


def test_an_anomaly_result_can_be_dumped_for_cross_tree_comparison():
    """Neither tree serialised this type, which is how the drift went unnoticed."""
    assert "serializeAnomaly" in _read(CPP / "common/utility/verify_serialize.hpp")
    assert '"AnomalyResult"' in _read(PY / "common/runner/verify_serialize.py")
