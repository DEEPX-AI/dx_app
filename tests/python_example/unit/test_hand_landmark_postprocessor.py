# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""MediaPipe HandLandmarkLite: z is depth, the hand's score is its presence.

HandLandmarkLite returns [1, 63] landmarks (x, y, z per point; z is relative
depth, negative toward the camera), [1, 1] hand presence, [1, 1] handedness and
[1, 63] world landmarks. The C++ postprocessor once stored z in each keypoint's
confidence, so the graph's keypoint drawing, which skips points below a score,
dropped all 21. The Python result has a z column, so here z stays depth and the
score lives in ``confidence`` - the same split the C++ side now makes (each
C++ keypoint's confidence is the presence score).
"""
from __future__ import annotations

import numpy as np

from common.base import PreprocessContext
from common.processors.hand_landmark_postprocessor import HandLandmarkPostprocessor
from common.runner.verify_serialize import _ser_hand_landmark
from common.visualizers import HandLandmarkVisualizer


def _outputs(presence=0.93, handedness=0.9):
    i = np.arange(21, dtype=np.float32)
    landmarks = np.stack([20.0 + i * 8.0, 200.0 - i * 7.0, -0.5 - i], axis=1)
    return [
        landmarks.reshape(1, 63),
        np.array([[presence]], dtype=np.float32),
        np.array([[handedness]], dtype=np.float32),
        np.full((1, 63), 0.01, dtype=np.float32),
    ]


def _ctx():
    ctx = PreprocessContext()
    ctx.original_width = 224
    ctx.original_height = 224
    return ctx


def _decode(outputs):
    return HandLandmarkPostprocessor(224, 224).process(outputs, _ctx())


def test_presence_is_the_hand_score_and_z_stays_depth():
    hands = _decode(_outputs())
    assert len(hands) == 1
    hand = hands[0]
    assert hand.landmarks.shape == (21, 3)
    assert 0.0 <= hand.confidence <= 1.0
    assert hand.confidence == np.float32(0.93)
    assert hand.handedness == "Right"
    np.testing.assert_allclose(hand.landmarks[:, 2], -0.5 - np.arange(21))
    assert tuple(hand.landmarks[20, :2]) == (180.0, 60.0)


def test_without_a_presence_output_the_hand_is_fully_confident():
    hands = _decode(_outputs()[:1])
    assert len(hands) == 1
    assert hands[0].confidence == 1.0
    assert hands[0].handedness == "Unknown"


def test_a_hand_below_the_threshold_is_dropped():
    assert _decode(_outputs(presence=0.2)) == []


def test_the_dump_reports_depth_as_z():
    dump = _ser_hand_landmark(_decode(_outputs()), 224, 224)
    point = dump["detections"][0]["landmarks"][3]
    assert point == {"x": 44.0, "y": 179.0, "z": -3.5}
    assert dump["detections"][0]["confidence"] == np.float32(0.93)


def test_the_visualizer_draws_every_landmark():
    hands = _decode(_outputs())
    canvas = HandLandmarkVisualizer().visualize(np.zeros((224, 224, 3), np.uint8), hands)
    for x, y in hands[0].landmarks[:, :2].astype(int):
        assert canvas[y - 2:y + 3, x - 2:x + 3].max() > 100, (x, y)
