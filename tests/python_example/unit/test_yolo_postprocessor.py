"""Tests for YOLO postprocess optimizations."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest

from common.processors.yolo_postprocessor import _limit_nms_candidates


ROOT = Path(__file__).resolve().parents[3]


def test_limit_nms_candidates_keeps_highest_yolo_scores() -> None:
    boxes = np.arange(20, dtype=np.float32).reshape(5, 4)
    scores = np.array([0.2, 0.95, 0.4, 0.8, 0.1], dtype=np.float32)
    class_ids = np.arange(5, dtype=np.int32)

    limited_boxes, limited_scores, limited_classes = _limit_nms_candidates(boxes, scores, class_ids, 3)

    assert limited_scores.tolist() == pytest.approx([0.95, 0.8, 0.4])
    assert limited_classes.tolist() == [1, 3, 2]
    assert limited_boxes.tolist() == boxes[[1, 3, 2]].tolist()


def test_limit_nms_candidates_returns_original_arrays_when_disabled() -> None:
    boxes = np.arange(20, dtype=np.float32).reshape(5, 4)
    scores = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    class_ids = np.arange(5, dtype=np.int32)

    limited_boxes, limited_scores, limited_classes = _limit_nms_candidates(boxes, scores, class_ids, 0)

    assert limited_boxes is boxes
    assert limited_scores is scores
    assert limited_classes is class_ids


def test_high_cost_yolo_configs_bound_nms_candidates() -> None:
    config_paths = [
        ROOT / 'src/python_example/object_detection/yolov5s_wo_spp_640/config.json',
    ]

    for config_path in config_paths:
        config = json.loads(config_path.read_text())
        assert config['max_nms_candidates'] <= 300, config_path
