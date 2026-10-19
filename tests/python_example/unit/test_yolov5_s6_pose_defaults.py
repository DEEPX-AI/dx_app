"""yolov5-s6-pose: the C++ factory's default thresholds equal the Python ones.

The variant's ``config.json`` has an empty ``config`` object, so neither runner
overrides the thresholds: the C++ ``Yolov5PoseFactory`` constructor defaults and
the Python ``YOLOv5PosePostprocessor`` defaults are what each language actually
runs with. A mismatch makes the two examples of the same model report different
poses on the same image without any error.

The C++ side is read from source (the constructor's default arguments); the
Python side is the postprocessor the Python factory really builds.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "yolov5-s6-pose_640x640"
REL_DIR = Path("pose_estimation") / "yolov5_pose" / VARIANT
CPP_FACTORY = (PROJECT_ROOT / "src" / "cpp_example" / REL_DIR / "factory"
               / f"{VARIANT}_factory.hpp")
PY_VARIANT_DIR = PROJECT_ROOT / "src" / "python_example" / REL_DIR
THRESHOLDS = ("obj_threshold", "score_threshold", "nms_threshold")


def _cpp_constructor_defaults(source: str) -> dict:
    """``{name: default}`` from ``Yolov5PoseFactory(float x = 0.25f, ...)``."""
    match = re.search(r"\bYolov5PoseFactory\s*\(([^)]*)\)", source)
    assert match, f"no Yolov5PoseFactory constructor in {CPP_FACTORY}"
    defaults = dict(
        (name, float(value)) for name, value in re.findall(
            r"float\s+(\w+)\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)f?",
            match.group(1)))
    return defaults


def _python_postprocessor():
    path = PY_VARIANT_DIR / "factory" / f"{VARIANT}_factory.py"
    spec = importlib.util.spec_from_file_location("yolov5_s6_pose_factory", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    factory = module.Yolov5PoseFactory(variant=VARIANT)
    return factory.create_postprocessor(640, 640)


def test_the_variant_config_overrides_no_threshold():
    """Otherwise the defaults below are not what either runner uses."""
    config = json.loads((PY_VARIANT_DIR / "config.json").read_text(encoding="utf-8"))
    overridden = sorted(set(config.get("config") or {}) & set(THRESHOLDS))
    assert not overridden, f"{VARIANT}/config.json overrides {overridden}"


def test_the_cpp_constructor_declares_every_threshold_default():
    defaults = _cpp_constructor_defaults(CPP_FACTORY.read_text(encoding="utf-8"))
    assert set(defaults) == set(THRESHOLDS), defaults


@pytest.mark.parametrize("name", THRESHOLDS)
def test_cpp_default_equals_python_default(name):
    cpp = _cpp_constructor_defaults(CPP_FACTORY.read_text(encoding="utf-8"))[name]
    python = float(getattr(_python_postprocessor(), name))
    assert cpp == pytest.approx(python, abs=1e-9), (
        f"{VARIANT} {name}: C++ factory default {cpp} != Python postprocessor "
        f"default {python}")
