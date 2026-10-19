# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""``--variant`` must resolve to exactly one model, from the registry alone.

The variant key IS the .dxnn stem (Phase 0), so resolution is a dict lookup, not a
normalisation heuristic. This test pins that: a known variant resolves to its own
.dxnn and config, a legacy example-dir name still resolves, and an unknown key raises
rather than silently falling back -- the failure mode the 4-pass fuzzy matcher had.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.variants import VariantNotFound, list_variants, resolve_variant  # noqa: E402


def test_known_variant_resolves_to_its_own_dxnn():
    v = resolve_variant("yolov5-s_640x640")
    assert v.dxnn_file == "yolov5-s_640x640.dxnn"
    assert v.task == "object_detection"
    assert v.family == "yolov5"
    assert (v.input_width, v.input_height) == (640, 640)
    assert v.postprocessor == "yolov5"


def test_variant_carries_the_task_config():
    v = resolve_variant("yolo26-n_640x640")
    assert v.config.get("score_threshold") is not None
    assert v.image_only is False


def test_image_only_variant_is_flagged():
    v = resolve_variant("arcface_mobilefacenet_112x112")
    assert v.image_only is True


def test_model_path_points_into_assets_models():
    v = resolve_variant("yolov5-s_640x640")
    assert v.model_path.parent == PROJECT_ROOT / "assets" / "models"
    assert v.model_path.name == "yolov5-s_640x640.dxnn"


def test_unknown_variant_raises_with_a_useful_message():
    with pytest.raises(VariantNotFound) as ei:
        resolve_variant("yolov5-s_640x640_typo")
    assert "yolov5-s_640x640_typo" in str(ei.value)


def test_lookup_by_legacy_model_name_also_works():
    """Legacy example-dir names stay resolvable so old scripts keep running."""
    assert resolve_variant("yolov5s").variant == "yolov5-s_640x640"
    assert resolve_variant("deit_base384_distilled").variant == "deit-b_384x384_distilled"


def test_list_variants_filters_by_family():
    got = {v.variant for v in list_variants(task="object_detection", family="yolov5")}
    assert "yolov5-s_640x640" in got
    assert "yolov5-m6_1280x1280_v6.1" in got
    assert all(v.startswith("yolov5") for v in got)


def test_list_variants_yields_every_distinct_variant():
    reg = json.loads((PROJECT_ROOT / "config" / "model_registry.json").read_text())
    assert {v.variant for v in list_variants()} == {e["variant"] for e in reg}


def test_every_registry_entry_is_resolvable():
    reg = json.loads((PROJECT_ROOT / "config" / "model_registry.json").read_text())
    for e in reg:
        assert resolve_variant(e["variant"]).dxnn_file == e["dxnn_file"]
        assert resolve_variant(e["model_name"]).dxnn_file == e["dxnn_file"]


def test_parse_common_args_accepts_variant_and_fills_model(monkeypatch):
    """``--variant`` populates ``args.model`` so runners consume args.model only."""
    from common.runner.args import parse_common_args

    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--variant", "yolov5-s_640x640", "--image", "sample/img/sample_street.jpg"],
    )
    args = parse_common_args("test")
    assert args.variant == "yolov5-s_640x640"
    assert args.model is not None
    assert args.model.replace("\\", "/").endswith("assets/models/yolov5-s_640x640.dxnn")


def test_explicit_model_wins_over_variant(monkeypatch):
    """--model is the escape hatch for a locally compiled .dxnn."""
    from common.runner.args import parse_common_args

    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--variant", "yolov5-s_640x640", "--model", "/tmp/other.dxnn",
         "--image", "sample/img/sample_street.jpg"],
    )
    args = parse_common_args("test")
    assert args.model == "/tmp/other.dxnn"


def test_unknown_variant_on_the_cli_fails_loudly(monkeypatch):
    from common.runner.args import parse_common_args

    monkeypatch.setattr(sys, "argv", ["prog", "--variant", "nope_123"])
    with pytest.raises(VariantNotFound):
        parse_common_args("test")


def test_omitting_variant_leaves_model_untouched(monkeypatch):
    """The pre-existing no-flags path must behave exactly as before."""
    from common.runner.args import parse_common_args

    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_common_args("test")
    assert args.variant is None
    assert args.model is None
