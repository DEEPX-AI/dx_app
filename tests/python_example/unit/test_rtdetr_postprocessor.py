# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""RT-DETR decode, written against the architecture because no model exists to measure.

None of the 20 RT-DETR / mask-RT-DETR additions has a published .dxnn or .onnx (every
URL returns 403), so this decode cannot be pinned to a measurement the way the
pre-optimized one was. It is instead written to the two output layouts PaddleDetection
actually produces, and to fail loudly on anything else:

  split      boxes  (1, N, 4)   cxcywh normalised to [0,1]
             logits (1, N, C)   per-class, before sigmoid
             -- the raw decoder head, which is what an export without the
                postprocess op emits.

  paddle_nms bbox   (1, N, 6)   [class_id, score, x1, y1, x2, y2]
             -- PaddleDetection's own DETRPostProcess, which is NMS-free: it
                sigmoids the logits, takes the top-k over the flattened
                (query x class) scores and converts to corners. An export that
                includes it emits this instead.

RT-DETR is NMS-free either way, and its preprocessing is a plain resize with no
letterbox, so ctx carries scale_x / scale_y and pad 0.

The layout is chosen by shape, and the ambiguity is the point: a (1, 300, 6) tensor
looks exactly like a pre-optimized row table but orders its columns differently
([class, score, box] vs [box, score, class]). Guessing wrong there produces
plausible-looking nonsense, so the variant config can force the layout and the error
message names the key.
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
    MaskRTDETRPostprocessor,
    RTDETRPostprocessor,
)

QUERIES = 300
CLASSES = 80


def _ctx(orig_w: int = 1280, orig_h: int = 720) -> PreprocessContext:
    """Plain-resize context, as SimpleResizePreprocessor builds it."""
    return PreprocessContext(
        original_width=orig_w, original_height=orig_h,
        input_width=640, input_height=640,
        scale_x=640.0 / orig_w, scale_y=640.0 / orig_h,
        scale=min(640.0 / orig_w, 640.0 / orig_h),
        pad_x=0, pad_y=0,
    )


def _split(rows: list[tuple[list[float], int, float]]):
    """(boxes, logits) in the raw-decoder layout. *rows* is [(cxcywh, class, logit)]."""
    boxes = np.zeros((1, QUERIES, 4), dtype=np.float32)
    logits = np.full((1, QUERIES, CLASSES), -10.0, dtype=np.float32)
    for i, (box, class_id, logit) in enumerate(rows):
        boxes[0, i] = box
        logits[0, i, class_id] = logit
    return boxes, logits


def _paddle(rows: list[list[float]], batched: bool = False):
    """PaddleDetection's [class, score, x1, y1, x2, y2] table.

    MEASURED on DX-RT 3.5.0: rtdetr-r18vd-6x_640x640 emits (300, 6) -- TWO dimensions,
    no batch axis -- with the corners in model-input pixels. mask-rtdetr emits (100, 6)
    the same way. `batched=True` produces the 3-D form as well, because nothing
    guarantees every compile drops the batch axis and a decode that only handles one of
    them is a decode that breaks on the next model.
    """
    table = np.zeros((QUERIES, 6), dtype=np.float32)
    for i, row in enumerate(rows):
        table[i] = row
    return table[None, ...] if batched else table


# --------------------------------------------------------------- split layout

def test_decodes_normalised_cxcywh_against_the_original_image():
    boxes, logits = _split([([0.5, 0.5, 0.25, 0.5], 5, 10.0)])
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                  ).process([boxes, logits], _ctx())
    assert len(results) == 1
    det = results[0]
    assert det.class_id == 5
    assert det.confidence == pytest.approx(1.0, abs=1e-3)
    assert det.box[0] == pytest.approx(1280 * 0.375, abs=0.5)   # (0.5 - 0.125) * W
    assert det.box[2] == pytest.approx(1280 * 0.625, abs=0.5)
    assert det.box[1] == pytest.approx(720 * 0.25, abs=0.5)     # (0.5 - 0.25) * H
    assert det.box[3] == pytest.approx(720 * 0.75, abs=0.5)


def test_tensor_order_does_not_matter():
    boxes, logits = _split([([0.5, 0.5, 0.2, 0.2], 1, 10.0)])
    post = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5})
    forward = post.process([boxes, logits], _ctx())
    reversed_ = post.process([logits, boxes], _ctx())
    assert len(forward) == len(reversed_) == 1
    assert forward[0].class_id == reversed_[0].class_id == 1
    assert forward[0].box == pytest.approx(reversed_[0].box)


def test_applies_no_nms_because_the_model_is_nms_free():
    """Two overlapping queries of different classes must both survive."""
    boxes, logits = _split([([0.5, 0.5, 0.3, 0.3], 2, 10.0),
                            ([0.5, 0.5, 0.3, 0.3], 7, 10.0)])
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                  ).process([boxes, logits], _ctx())
    assert {d.class_id for d in results} == {2, 7}


def test_nms_can_be_switched_on_for_a_checkpoint_that_needs_it():
    boxes, logits = _split([([0.5, 0.5, 0.3, 0.3], 2, 10.0),
                            ([0.5, 0.5, 0.3, 0.3], 7, 9.0)])
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5,
                                             "nms_threshold": 0.45}
                                  ).process([boxes, logits], _ctx())
    assert len(results) == 1


def test_already_sigmoided_scores_are_not_squashed_twice():
    """A model compiled with the sigmoid folded in emits probabilities already."""
    boxes = np.zeros((1, QUERIES, 4), dtype=np.float32)
    scores = np.zeros((1, QUERIES, CLASSES), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.2, 0.2]
    scores[0, 0, 3] = 0.93
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.9}
                                  ).process([boxes, scores], _ctx())
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.93)


def test_results_are_score_descending():
    boxes, logits = _split([([0.2, 0.2, 0.1, 0.1], 0, 1.0),
                            ([0.5, 0.5, 0.1, 0.1], 1, 5.0),
                            ([0.8, 0.8, 0.1, 0.1], 2, 3.0)])
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                  ).process([boxes, logits], _ctx())
    scores = [d.confidence for d in results]
    assert scores == sorted(scores, reverse=True)


def test_boxes_are_clipped_into_the_frame():
    boxes, logits = _split([([0.05, 0.05, 0.4, 0.4], 0, 10.0)])   # runs off the edge
    det = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                              ).process([boxes, logits], _ctx())[0]
    assert det.box[0] >= 0.0 and det.box[1] >= 0.0
    assert det.box[2] <= 1280.0 and det.box[3] <= 720.0


# --------------------------------------------------------------- paddle layout

def test_decodes_the_paddle_postprocess_layout():
    """[class, score, x1, y1, x2, y2] in model-input pixels -- the measured layout."""
    table = _paddle([[5.0, 0.9, 64.0, 128.0, 192.0, 320.0]])
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5,
                                             "layout": "paddle_nms"}
                                  ).process([table], _ctx())
    assert len(results) == 1
    det = results[0]
    assert det.class_id == 5
    assert det.confidence == pytest.approx(0.9)
    # scale_x = 640/1280 = 0.5, scale_y = 640/720
    assert det.box[0] == pytest.approx(128.0, abs=0.5)
    assert det.box[2] == pytest.approx(384.0, abs=0.5)
    assert det.box[1] == pytest.approx(128.0 / (640.0 / 720.0), abs=0.5)


def test_the_paddle_layout_is_detected_from_a_lone_six_column_tensor():
    table = _paddle([[5.0, 0.9, 64.0, 128.0, 192.0, 320.0]])
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                  ).process([table], _ctx())
    assert len(results) == 1 and results[0].class_id == 5


def test_the_batched_paddle_layout_decodes_identically():
    """(1, N, 6) and (N, 6) must give the same answer."""
    rows = [[5.0, 0.9, 64.0, 128.0, 192.0, 320.0]]
    post = RTDETRPostprocessor(640, 640, {"score_threshold": 0.5})
    flat = post.process([_paddle(rows)], _ctx())
    batched = post.process([_paddle(rows, batched=True)], _ctx())
    assert len(flat) == len(batched) == 1
    assert flat[0].class_id == batched[0].class_id
    assert flat[0].box == pytest.approx(batched[0].box)


def test_paddle_layout_padding_rows_are_ignored():
    table = _paddle([[5.0, 0.9, 64.0, 128.0, 192.0, 320.0]])   # 299 zero rows follow
    results = RTDETRPostprocessor(640, 640, {"score_threshold": 0.0,
                                             "layout": "paddle_nms"}
                                  ).process([table], _ctx())
    assert len(results) == 1


# --------------------------------------------------------------- failure modes

def test_unrecognisable_outputs_raise_with_the_actual_shapes():
    junk = [np.zeros((1, 25200, 85), dtype=np.float32)]
    with pytest.raises(ValueError) as exc:
        RTDETRPostprocessor(640, 640).process(junk, _ctx())
    message = str(exc.value)
    assert "(1, 25200, 85)" in message
    assert "layout" in message, "the error must name the config key that forces a layout"


def test_an_explicit_layout_that_does_not_fit_fails_rather_than_falling_back():
    boxes, logits = _split([([0.5, 0.5, 0.2, 0.2], 1, 10.0)])
    with pytest.raises(ValueError) as exc:
        RTDETRPostprocessor(640, 640, {"layout": "paddle_nms"}
                            ).process([boxes, logits], _ctx())
    assert "paddle_nms" in str(exc.value)


# --------------------------------------------------------------- mask variant

def test_mask_rtdetr_pairs_each_query_with_its_measured_mask():
    """MEASURED: mask-rtdetr emits (100, 6) + (100, 640, 640) int32 0/1.

    The masks are ALREADY binary and already at the model input resolution -- not
    logits, not prototypes, no sigmoid to apply. The first version of this decode
    expected (1, N, H, W) float logits and would have squashed an integer 1 to 0.73
    before thresholding it.
    """
    table = _paddle([[3.0, 0.9, 128.0, 128.0, 384.0, 384.0]])
    masks = np.zeros((QUERIES, 640, 640), dtype=np.int32)
    masks[0, 100:500, 100:500] = 1
    results = MaskRTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                      ).process([table, masks], _ctx())
    assert len(results) == 1
    assert results[0].class_id == 3
    assert results[0].mask.shape == (720, 1280)
    assert results[0].mask.dtype == np.uint8
    assert results[0].mask.sum() > 0


def test_mask_rtdetr_accepts_float_logit_masks_too():
    """A compile that leaves the masks as logits must still decode."""
    table = _paddle([[3.0, 0.9, 128.0, 128.0, 384.0, 384.0]])
    masks = np.full((QUERIES, 160, 160), -10.0, dtype=np.float32)
    masks[0] = 10.0
    results = MaskRTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                      ).process([table, masks], _ctx())
    assert len(results) == 1 and results[0].mask.sum() > 0


def test_mask_rtdetr_confines_the_mask_to_its_box():
    # box in input pixels 256..384 x 256..320 -> original 512..768 x 288..360
    table = _paddle([[3.0, 0.9, 256.0, 256.0, 384.0, 320.0]])
    masks = np.ones((QUERIES, 640, 640), dtype=np.int32)   # on everywhere
    mask = MaskRTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                   ).process([table, masks], _ctx())[0].mask
    assert mask[:200, :].sum() == 0, "painted above the box"
    assert mask[:, :400].sum() == 0, "painted left of the box"
    assert mask[300:355, 530:750].sum() > 0, "box interior is empty"


def test_mask_rtdetr_says_so_when_the_mask_tensor_is_missing():
    table = _paddle([[3.0, 0.9, 128.0, 128.0, 384.0, 384.0]])
    with pytest.raises(ValueError) as exc:
        MaskRTDETRPostprocessor(640, 640, {"score_threshold": 0.5}
                                ).process([table], _ctx())
    message = str(exc.value)
    assert "mask" in message.lower()
    assert "(300, 6)" in message, "the error must print the actual shapes"
