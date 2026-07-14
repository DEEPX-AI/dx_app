"""Tests for EfficientDet postprocess optimizations."""

from __future__ import annotations

import numpy as np
import pytest

from common.processors.efficientdet_postprocessor import EfficientDetPostprocessor


def test_limit_nms_candidates_keeps_highest_scores() -> None:
    post = EfficientDetPostprocessor(640, 640, {"max_nms_candidates": 3})
    boxes = np.arange(20, dtype=np.float32).reshape(5, 4)
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2], dtype=np.float32)
    class_ids = np.arange(5, dtype=np.int32)

    limited_boxes, limited_scores, limited_classes, _ = post._limit_nms_candidates(boxes, scores, class_ids)

    assert limited_scores.tolist() == pytest.approx([0.9, 0.8, 0.3])
    assert limited_classes.tolist() == [1, 3, 2]
    assert limited_boxes.tolist() == boxes[[1, 3, 2]].tolist()


def test_limit_nms_candidates_can_be_disabled() -> None:
    post = EfficientDetPostprocessor(640, 640, {"max_nms_candidates": 0})
    boxes = np.arange(20, dtype=np.float32).reshape(5, 4)
    scores = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    class_ids = np.arange(5, dtype=np.int32)

    limited_boxes, limited_scores, limited_classes, _ = post._limit_nms_candidates(boxes, scores, class_ids)

    assert limited_boxes is boxes
    assert limited_scores is scores
    assert limited_classes is class_ids
