# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The palm detector must see the frame at its own aspect ratio.

MediaPipe's palm detector is trained on aspect-preserving (letterboxed) input. The
example stretch-resized a 16:9 frame to 192x192 and mapped the normalized box back with
``n * original``. On assets/videos/hand.mp4 that lost the two hands at 1 s, put the box
beside the fist at 5 s and boxed two fingers at 7 s; padding the same frames to a square
found all three (0.59 / 0.65 / 0.93). The square palm->hand expansion (box_scale) was also
skewed into a wide rectangle by the stretch.

So: the preprocessor letterboxes (pad 0, the MediaPipe convention) in both trees, and the
box is mapped back with one formula that is right for either preprocessor --
stretch (scale_x != scale_y): x = n * W_in / scale_x; letterbox: x = (n * W_in - pad_x) / scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.base import PreprocessContext
from common.processors.mediapipe_hand_postprocessor import _to_original

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "hand_detection/mediapipe_hand_detector/mediapipe-hand-detector_192x192"


def _ctx(**kw):
    ctx = PreprocessContext()
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def test_a_letterboxed_point_maps_back_to_the_frame():
    # 1280x720 -> 192x108 + 42 px top/bottom pad
    ctx = _ctx(original_width=1280, original_height=720, input_width=192, input_height=192,
               scale=0.15, pad_x=0, pad_y=42)
    x, y = _to_original(np.array([0.5, 0.0]), np.array([0.5, 42 / 192]), ctx, 192)
    assert np.allclose(x, [640, 0]) and np.allclose(y, [360, 0])


def test_a_square_box_stays_square_when_letterboxed():
    ctx = _ctx(original_width=1280, original_height=720, input_width=192, input_height=192,
               scale=0.15, pad_x=0, pad_y=42)
    x, y = _to_original(np.array([0.4, 0.6]), np.array([0.4, 0.6]), ctx, 192)
    assert abs((x[1] - x[0]) - (y[1] - y[0])) < 1e-6


def test_a_stretched_point_still_maps_back():
    ctx = _ctx(original_width=1280, original_height=720, input_width=192, input_height=192,
               scale=0.15, scale_x=192 / 1280, scale_y=192 / 720, pad_x=0, pad_y=0)
    x, y = _to_original(np.array([0.5, 1.0]), np.array([0.5, 1.0]), ctx, 192)
    assert np.allclose(x, [640, 1280]) and np.allclose(y, [360, 720])


def test_both_trees_letterbox_the_palm_detector():
    for tree in ("python_example", "cpp_example"):
        spec = json.loads((PROJECT_ROOT / "src" / tree / VARIANT / "config.json").read_text())["preprocessor"]
        assert spec["class"] == "LetterboxPreprocessor", (tree, spec["class"])
        assert spec["kwargs"].get("pad_color") == [0, 0, 0], tree
    factory = (PROJECT_ROOT / "src/cpp_example" / VARIANT / "factory/mediapipe-hand-detector_192x192_factory.hpp")
    src = factory.read_text()
    assert "DetectionPreprocessor" in src and "SimpleResizePreprocessor>" not in src
