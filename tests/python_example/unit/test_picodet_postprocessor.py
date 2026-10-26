# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""PP-PicoDet decode, pinned to what PP-ShiTu's mainbody detector actually emits.

MEASURED on DX-RT 3.5.0, pp-shituv1-mainbody-detection_640x640 at 640x640:

    (1, 6400, 1)  (1, 1600, 1)  (1, 400, 1)  (1, 100, 1)     class scores, 1 class
    (1, 6400, 32) (1, 1600, 32) (1, 400, 32) (1, 100, 32)    box distributions

Four levels, anchor counts 6400/1600/400/100 -> strides 8/16/32/64 at 640, and
32 = 4 sides x (reg_max + 1) with reg_max = 7. The scores are ALREADY sigmoid
(measured range 0.0005..0.2615) and the distributions are NOT yet softmax (measured
per-side sums around 0, not 1).

That is why NanoDetPostprocessor -- the first choice here, because PicoDet and
NanoDet-Plus share the GFL head -- does not fit: it expects ONE concatenated tensor
of shape (1, N, num_classes + 4*(reg_max+1)), and this model emits eight separate
ones. Same head, different packaging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import PicoDetPostprocessor  # noqa: E402

STRIDES = (8, 16, 32, 64)
REG_MAX = 7
INPUT = 640


def _ctx(orig_w: int = 1280, orig_h: int = 640) -> PreprocessContext:
    """Letterbox context: 1280x640 into 640x640 -> scale 0.5, pad_y 160."""
    scale = min(INPUT / orig_w, INPUT / orig_h)
    return PreprocessContext(
        original_width=orig_w, original_height=orig_h,
        input_width=INPUT, input_height=INPUT, scale=scale,
        pad_x=(INPUT - orig_w * scale) / 2.0,
        pad_y=(INPUT - orig_h * scale) / 2.0,
    )


def _levels(num_classes: int = 1):
    """Empty per-level tensors in the measured layout, scores first then boxes."""
    scores, boxes = [], []
    for stride in STRIDES:
        n = (INPUT // stride) ** 2
        scores.append(np.zeros((1, n, num_classes), dtype=np.float32))
        boxes.append(np.zeros((1, n, 4 * (REG_MAX + 1)), dtype=np.float32))
    return scores, boxes


def _put(scores, boxes, level: int, cell: int, score: float, distance: float,
         class_id: int = 0):
    """One anchor at *cell* on *level*, with every side at *distance* grid units."""
    scores[level][0, cell, class_id] = score
    # A one-hot distribution at bin `distance` integrates to exactly that distance.
    for side in range(4):
        boxes[level][0, cell, side * (REG_MAX + 1) + int(distance)] = 20.0
    return scores, boxes


def test_decodes_one_anchor_into_a_box_in_original_coordinates():
    scores, boxes = _levels()
    # level 0 is stride 8, an 80x80 grid; cell 3240 is (col 40, row 40) -> centre
    _put(scores, boxes, level=0, cell=40 * 80 + 40, score=0.9, distance=4)
    results = PicoDetPostprocessor(INPUT, INPUT, {"conf_threshold": 0.3}
                                   ).process(scores + boxes, _ctx())
    assert len(results) == 1
    det = results[0]
    assert det.class_id == 0
    assert det.confidence == pytest.approx(0.9, abs=1e-3)
    # centre (40.5, 40.5) * 8 = (324, 324); +-4 grid units * 8 = +-32 px
    # original = (input - pad) / scale, pad_x 0 pad_y 160, scale 0.5
    assert det.box[0] == pytest.approx((324 - 32) * 2, abs=2.0)
    assert det.box[1] == pytest.approx((324 - 32 - 160) * 2, abs=2.0)


def test_scores_below_the_threshold_are_dropped():
    scores, boxes = _levels()
    _put(scores, boxes, level=0, cell=100, score=0.2, distance=2)
    post = PicoDetPostprocessor(INPUT, INPUT, {"conf_threshold": 0.3})
    assert post.process(scores + boxes, _ctx()) == []


def test_every_level_is_decoded_not_just_the_first():
    """A decode that stops after level 0 would still look right on most images."""
    for level, stride in enumerate(STRIDES):
        scores, boxes = _levels()
        grid = INPUT // stride
        _put(scores, boxes, level=level, cell=(grid // 2) * grid + grid // 2,
             score=0.9, distance=3)
        results = PicoDetPostprocessor(INPUT, INPUT, {"conf_threshold": 0.3}
                                       ).process(scores + boxes, _ctx())
        assert len(results) == 1, f"level {level} (stride {stride}) was not decoded"


def test_already_sigmoided_scores_are_not_squashed_again():
    """MEASURED: this model's scores are already probabilities (0.0005..0.2615)."""
    scores, boxes = _levels()
    _put(scores, boxes, level=1, cell=200, score=0.75, distance=2)
    det = PicoDetPostprocessor(INPUT, INPUT, {"conf_threshold": 0.5}
                               ).process(scores + boxes, _ctx())[0]
    assert det.confidence == pytest.approx(0.75, abs=1e-3)


def test_overlapping_anchors_are_merged_by_nms():
    scores, boxes = _levels()
    centre = 40 * 80 + 40
    _put(scores, boxes, level=0, cell=centre, score=0.9, distance=4)
    _put(scores, boxes, level=0, cell=centre + 1, score=0.8, distance=4)
    results = PicoDetPostprocessor(INPUT, INPUT, {"conf_threshold": 0.3,
                                                  "nms_threshold": 0.5}
                                   ).process(scores + boxes, _ctx())
    assert len(results) == 1


def test_a_single_concatenated_tensor_is_refused_with_its_shape():
    """That is NanoDet's packaging, and using this decode on it would be wrong."""
    combined = [np.zeros((1, 8400, 1 + 4 * (REG_MAX + 1)), dtype=np.float32)]
    with pytest.raises(ValueError) as exc:
        PicoDetPostprocessor(INPUT, INPUT).process(combined, _ctx())
    message = str(exc.value)
    assert "(1, 8400, 33)" in message
    assert "level" in message.lower()


def test_boxes_stay_inside_the_original_frame():
    scores, boxes = _levels()
    _put(scores, boxes, level=0, cell=0, score=0.9, distance=7)   # top-left corner
    det = PicoDetPostprocessor(INPUT, INPUT, {"conf_threshold": 0.3}
                               ).process(scores + boxes, _ctx())[0]
    assert det.box[0] >= 0.0 and det.box[1] >= 0.0
    assert det.box[2] <= 1280.0 and det.box[3] <= 640.0
