"""Family-level Python scripts start, and resolve a model without -m / --variant.

Under the task/family/variant layout every example entry point lives in its
variant directory; the scripts left directly on a family directory are the
``<model>_<kind>_ort_off.py`` debug scripts (ONNX Runtime off). The family has
no ``factory/`` of its own, so they could not even import their factory, and
without ``-m`` their name (``yolov8n_sync_ort_off``) matched no registry row.

They now run the registry variant of the model they are named after
(``yolov8n`` -> ``yolov8-n_640x640``), falling back to the family's default
variant (the first published one) for a name the registry does not know; the
default model without ``-m`` is that variant's ``.dxnn``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from test_helpers.proc import example_python, run_bounded
from common.runner.entry import ort_off_variant_dir
from common.runner.sync_runner import _resolve_default_model_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = PROJECT_ROOT / "src" / "python_example"
REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"

FAMILY_SCRIPTS = sorted(
    path for path in PY_ROOT.glob("*/*/*.py")
    if path.parts[-3] != "common" and ("_sync" in path.stem or "_async" in path.stem)
)


def _dxnn_of(variant: str) -> str:
    rows = json.loads(REGISTRY.read_text(encoding="utf-8"))
    row = next(r for r in rows if r["variant"] == variant and not r.get("alias_of"))
    return "assets/models/" + row["dxnn_file"]


def test_the_family_level_scripts_are_found():
    assert FAMILY_SCRIPTS, "no family-level *_sync / *_async script found"


@pytest.mark.parametrize("script", FAMILY_SCRIPTS, ids=lambda p: p.name)
def test_a_family_script_runs_a_variant_that_has_a_factory(script):
    variant_dir = ort_off_variant_dir(script)
    assert variant_dir.parent == script.parent, variant_dir
    assert (variant_dir / "factory" / "__init__.py").is_file(), variant_dir


@pytest.mark.parametrize("script", FAMILY_SCRIPTS, ids=lambda p: p.name)
def test_a_family_script_without_m_resolves_its_variants_model(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert _resolve_default_model_path() == _dxnn_of(ort_off_variant_dir(script).name)


def test_a_script_named_after_a_model_runs_that_models_variant():
    script = PY_ROOT / "object_detection" / "yolov8" / "yolov8n_sync_ort_off.py"
    assert ort_off_variant_dir(script).name == "yolov8-n_640x640"


def test_an_unknown_name_falls_back_to_the_family_default(tmp_path):
    from common.variant_config import default_variant

    family = PY_ROOT / "object_detection" / "yolov8"
    script = family / "no-such-model_sync_ort_off.py"      # never created
    assert ort_off_variant_dir(script) == family / default_variant(str(family))


@pytest.mark.parametrize("script", FAMILY_SCRIPTS, ids=lambda p: p.name)
def test_a_family_script_starts(script):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = run_bounded([example_python(), str(script), "--help"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--image" in result.stdout
