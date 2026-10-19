# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Pre-optimized decode, pinned to the contract measured on the NPU.

dx_yolo26 ships three real pre-optimized .dxnn files. Run through dx_engine on an M1
(DXRT v3.4.2) they produce exactly:

    pre_optimized_yolo26-n-od.dxnn     (1, 300,  6)
    pre_optimized_yolo26n-pose.dxnn    (1, 300, 57)
    pre_optimized_yolo26n-seg.dxnn     (1, 300, 38) + (1, 32, 160, 160)

A row is [x1, y1, x2, y2, score, class_id, extra...] in letterboxed input pixels,
score-descending. The tests below use synthetic tables of that exact shape, so the
decode is pinned without needing the NPU. The hardware side -- that the NPU really
does emit this contract -- is scripts/verify_preopt_contract.py, driven by
tests/python_example/test_preopt_contract.py under the `e2e` marker, because this
directory's conftest installs a Mock for dx_engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import (  # noqa: E402
    PreoptDetectionPostprocessor,
    PreoptPosePostprocessor,
    PreoptSegPostprocessor,
)

def _ctx(orig_w: int = 1280, orig_h: int = 720,
         input_w: int = 640, input_h: int = 640) -> PreprocessContext:
    """Letterbox context for a 1280x720 source scaled into 640x640.

    scale 0.5, pad_x 0, pad_y 140 -- so an input-pixel y of 100 maps to -80 in the
    original and must come back CLIPPED to 0, which is what every other
    postprocessor in this tree does.
    """
    scale = min(input_w / orig_w, input_h / orig_h)
    return PreprocessContext(
        original_width=orig_w, original_height=orig_h,
        input_width=input_w, input_height=input_h,
        scale=scale,
        pad_x=(input_w - orig_w * scale) / 2.0,
        pad_y=(input_h - orig_h * scale) / 2.0,
    )


def _table(rows: list[list[float]], cols: int) -> np.ndarray:
    """A fixed-size K=300 table with *rows* filled in and the rest zero."""
    table = np.zeros((1, 300, cols), dtype=np.float32)
    for i, row in enumerate(rows):
        table[0, i, :len(row)] = row
    return table


# --------------------------------------------------------------------- detection

def test_detection_keeps_rows_above_threshold_and_scales_to_original():
    post = PreoptDetectionPostprocessor(640, 640, {"score_threshold": 0.5,
                                                   "nms_threshold": 1.0})
    outputs = [_table([[100, 100, 200, 260, 0.90, 2.0],
                       [300, 300, 340, 340, 0.10, 3.0]], 6)]
    results = post.process(outputs, _ctx())
    assert len(results) == 1
    det = results[0]
    assert det.class_id == 2
    assert det.confidence == pytest.approx(0.90)
    assert det.box[0] == pytest.approx(200.0, abs=1e-3)
    assert det.box[1] == pytest.approx(0.0, abs=1e-3)      # -80 clipped
    assert det.box[2] == pytest.approx(400.0, abs=1e-3)
    assert det.box[3] == pytest.approx(240.0, abs=1e-3)


def test_detection_nms_merges_the_duplicate_class_rows():
    """The second top-k is over (anchor x class), so one box can appear twice."""
    post = PreoptDetectionPostprocessor(640, 640, {"score_threshold": 0.1,
                                                   "nms_threshold": 0.45})
    outputs = [_table([[100, 200, 200, 300, 0.90, 2.0],    # car
                       [101, 201, 201, 301, 0.80, 7.0]], 6)]  # same box as truck
    assert len(post.process(outputs, _ctx())) == 1


def test_detection_keeps_both_when_nms_is_disabled():
    post = PreoptDetectionPostprocessor(640, 640, {"score_threshold": 0.1,
                                                   "nms_threshold": 1.0})
    outputs = [_table([[100, 200, 200, 300, 0.90, 2.0],
                       [101, 201, 201, 301, 0.80, 7.0]], 6)]
    assert {d.class_id for d in post.process(outputs, _ctx())} == {2, 7}


def test_detection_ignores_the_all_zero_padding_rows():
    """K is fixed at 300; unused rows are zero and must not become boxes at the origin."""
    post = PreoptDetectionPostprocessor(640, 640, {"score_threshold": 0.3})
    assert len(post.process([_table([], 6)], _ctx())) == 0
    assert len(post.process([_table([[10, 200, 50, 260, 0.9, 1.0]], 6)], _ctx())) == 1


def test_a_zero_score_threshold_still_rejects_the_padding_rows():
    """A 0.0 threshold is the trap: >= 0.0 would admit all 300 zero rows."""
    post = PreoptDetectionPostprocessor(640, 640, {"score_threshold": 0.0,
                                                   "nms_threshold": 1.0})
    results = post.process([_table([[10, 200, 50, 260, 0.9, 1.0]], 6)], _ctx())
    assert len(results) == 1


def test_detection_results_are_score_descending():
    post = PreoptDetectionPostprocessor(640, 640, {"score_threshold": 0.1,
                                                   "nms_threshold": 1.0})
    outputs = [_table([[10, 200, 50, 260, 0.5, 0.0],
                       [100, 200, 140, 260, 0.9, 1.0],
                       [200, 200, 240, 260, 0.7, 2.0]], 6)]
    scores = [d.confidence for d in post.process(outputs, _ctx())]
    assert scores == sorted(scores, reverse=True)


def test_a_table_shaped_output_is_required_and_the_error_names_the_shapes():
    post = PreoptDetectionPostprocessor(640, 640)
    with pytest.raises(ValueError) as exc:
        post.process([np.zeros((1, 84, 8400), dtype=np.float32)], _ctx())
    message = str(exc.value)
    assert "(1, 84, 8400)" in message
    assert "pre-optimize" in message.lower()


# --------------------------------------------------------------------- pose

def test_pose_decodes_17_keypoints_and_clamps_into_the_image():
    post = PreoptPosePostprocessor(640, 640, {"score_threshold": 0.3,
                                              "nms_threshold": 1.0})
    row = [100, 200, 200, 300, 0.9, 0.0]
    for k in range(17):
        row += [120 + k, 210 + k, 0.8]
    results = post.process([_table([row], 57)], _ctx())
    assert len(results) == 1
    assert len(results[0].keypoints) == 17
    for keypoint in results[0].keypoints:
        assert 0.0 <= keypoint.x <= 1279.0
        assert 0.0 <= keypoint.y <= 719.0
        assert keypoint.confidence == pytest.approx(0.8)


def test_pose_rejects_a_table_too_narrow_for_its_keypoints():
    post = PreoptPosePostprocessor(640, 640, {"score_threshold": 0.3})
    with pytest.raises(ValueError) as exc:
        post.process([_table([[100, 200, 200, 300, 0.9, 0.0]], 6)], _ctx())
    assert "57" in str(exc.value)


# --------------------------------------------------------------------- segmentation

def test_seg_requires_the_prototype_tensor_and_says_so():
    post = PreoptSegPostprocessor(640, 640, {"score_threshold": 0.3})
    with pytest.raises(ValueError) as exc:
        post.process([_table([[100, 200, 300, 400, 0.9, 0.0]], 38)], _ctx())
    message = str(exc.value)
    assert "prototype" in message.lower()
    assert "(1, 300, 38)" in message, "the error must print the actual shapes"


def test_seg_produces_a_mask_at_original_resolution():
    post = PreoptSegPostprocessor(640, 640, {"score_threshold": 0.3,
                                             "nms_threshold": 1.0})
    row = [100, 200, 300, 400, 0.9, 0.0] + [0.1] * 32
    proto = np.ones((1, 32, 160, 160), dtype=np.float32)
    results = post.process([_table([row], 38), proto], _ctx())
    assert len(results) == 1
    assert results[0].mask.shape == (720, 1280)
    assert results[0].mask.dtype == np.uint8


def test_seg_mask_is_confined_to_its_box():
    """A coefficient set that activates everywhere must still not paint outside the box."""
    post = PreoptSegPostprocessor(640, 640, {"score_threshold": 0.3,
                                             "nms_threshold": 1.0})
    row = [200, 200, 400, 400, 0.9, 0.0] + [1.0] * 32
    proto = np.ones((1, 32, 160, 160), dtype=np.float32)
    mask = post.process([_table([row], 38), proto], _ctx())[0].mask
    # box in original coords: x 400..800, y 120..520
    assert mask[:100, :].sum() == 0, "painted above the box"
    assert mask[:, :300].sum() == 0, "painted left of the box"
    assert mask[200:400, 450:750].sum() > 0, "box interior is empty"


# The real-NPU check is NOT here: tests/python_example/conftest.py installs a Mock
# for dx_engine during pytest_configure, which is right for unit tests and fatal for a
# hardware one. It lives in scripts/verify_preopt_contract.py, run as a subprocess by
# tests/python_example/test_preopt_contract.py under the existing `e2e` marker -- the
# same shape every other real-NPU test in this tree uses.
