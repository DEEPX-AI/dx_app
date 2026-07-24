"""SFA3D postprocessor unit tests."""

import math

import numpy as np
import pytest

from common.base import DetectionResult
from common.processors import Detection3DResult, SFA3DPostprocessor
from common.utility import convert_cpp_sfa3d


def _make_outputs():
    return [
        np.zeros((1, 3, 152, 152), dtype=np.float32),
        np.zeros((1, 2, 152, 152), dtype=np.float32),
        np.zeros((1, 2, 152, 152), dtype=np.float32),
        np.zeros((1, 1, 152, 152), dtype=np.float32),
        np.zeros((1, 3, 152, 152), dtype=np.float32),
    ]


class TestSFA3DPostprocessor:
    def test_returns_empty_when_no_peak_above_threshold(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)  # every heatmap logit well below the score threshold

        results = postprocessor.process(outputs, ctx)

        assert results == []

    def test_decodes_single_peak_to_3d_detection(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        y, x = 10, 20
        outputs[0][0, 1, y, x] = 10.0   # class 1 (Car) heatmap peak
        outputs[2][0, 0, y, x] = 1.0    # direction: yaw_im
        outputs[2][0, 1, y, x] = 0.0    # direction: yaw_re
        outputs[3][0, 0, y, x] = 2.5    # z_coord (network regresses z - minZ)
        outputs[4][0, 0, y, x] = 1.5    # dim_h
        outputs[4][0, 1, y, x] = 8.0    # dim_w
        outputs[4][0, 2, y, x] = 16.0   # dim_l

        results = postprocessor.process(outputs, ctx)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, Detection3DResult)
        assert det.class_id == 1
        assert det.class_name == "Car"
        assert det.confidence > 0.99
        # BEV centre: (col/row) * (input / grid) with zero offset → 20*4, 10*4
        assert det.bev_x == pytest.approx(80.0)
        assert det.bev_y == pytest.approx(40.0)
        # dims are regressed directly in metres (no exp)
        assert (det.dim_h, det.dim_w, det.dim_l) == pytest.approx((1.5, 8.0, 16.0))
        # z regresses (z - minZ); minZ == -2.73
        assert det.z3d == pytest.approx(2.5 - 2.73)
        # yaw = atan2(im=1, re=0) == +pi/2
        assert det.yaw == pytest.approx(math.pi / 2.0)

    def test_decodes_negative_logit_peak_above_threshold(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        y, x = 10, 20
        outputs[0][0, 0, y, x] = -0.5   # class 0 (Pedestrian): sigmoid(-0.5) > 0.3 threshold
        outputs[4][0, 1, y, x] = 8.0
        outputs[4][0, 2, y, x] = 16.0

        results = postprocessor.process(outputs, ctx)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, Detection3DResult)
        assert det.class_id == 0
        assert det.class_name == "Pedestrian"
        assert det.confidence == pytest.approx(1.0 / (1.0 + math.exp(0.5)), rel=1e-6)
        assert det.bev_x == pytest.approx(80.0)
        assert det.bev_y == pytest.approx(40.0)


class TestConvertCppSFA3D:
    def test_convert_cpp_sfa3d_maps_rows_to_detection_results(self):
        detections = np.array(
            [
                [80.0, 40.0, 2.5, 1.5, 8.0, 16.0, math.pi / 2.0, 0.95, 2.0],
            ],
            dtype=np.float32,
        )

        results = convert_cpp_sfa3d(detections)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, DetectionResult)
        assert det.class_id == 2
        assert det.confidence == pytest.approx(0.95)
        np.testing.assert_allclose(det.box, [72.0, 36.0, 88.0, 44.0], atol=1e-6)

    def test_convert_cpp_sfa3d_keeps_model_space_boxes_with_ctx(self, ctx):
        detections = np.array(
            [
                [80.0, 40.0, 0.0, 1.5, 8.0, 16.0, 0.0, 0.95, 1.0],
            ],
            dtype=np.float32,
        )

        results = convert_cpp_sfa3d(detections, ctx)

        assert len(results) == 1
        np.testing.assert_allclose(results[0].box, [72.0, 36.0, 88.0, 44.0], atol=1e-6)
