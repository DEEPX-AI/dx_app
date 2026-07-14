import numpy as np

from common.processors.yolact_postprocessor import YOLACTPostprocessor


def test_limit_kept_detections_preserves_highest_score_order():
    post = YOLACTPostprocessor(512, 512, {'max_detections': 3})
    keep = np.array([4, 1, 8, 3, 2], dtype=int)

    limited = post._limit_kept_detections(keep)

    np.testing.assert_array_equal(limited, np.array([4, 1, 8]))


def test_limit_kept_detections_can_be_disabled():
    post = YOLACTPostprocessor(512, 512, {'max_detections': 0})
    keep = np.array([4, 1, 8, 3, 2], dtype=int)

    limited = post._limit_kept_detections(keep)

    assert limited is keep
