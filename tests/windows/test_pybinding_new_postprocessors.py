"""
New pybind postprocessors tests.
VitPose, DOPE, SuperPoint, YOLOPv2, MediaPipeHand
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if sys.platform == "win32":
    _DXRT_DIR = os.environ.get("DEEPX_SDK_DIR", r"C:\Program Files\DEEPX\DXNN\sdk")
    _DLL_SEARCH_DIRS = [
        os.path.join(_DXRT_DIR, "csharp"),
        os.path.join(_DXRT_DIR, "bin"),
        str(PROJECT_ROOT / "bin" / "Release"),
        str(PROJECT_ROOT / "lib"),
    ]
    for _dll_dir in _DLL_SEARCH_DIRS:
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)


def _try_import():
    try:
        import dx_postprocess
        return dx_postprocess
    except ImportError:
        return None


dx_postprocess = _try_import()
skip_if_no_binding = pytest.mark.skipif(
    dx_postprocess is None,
    reason="dx_postprocess not installed",
)


@skip_if_no_binding
class TestVitPosePostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.VitPosePostProcess(192, 256)
        assert proc.get_input_width() == 192
        assert proc.get_input_height() == 256

    @pytest.mark.pybinding
    def test_postprocess_shape_17x3(self):
        """Output shape must be [17, 3]"""
        proc = dx_postprocess.VitPosePostProcess(192, 256)
        heatmap = np.random.rand(1, 17, 4, 3).astype(np.float32)
        result = proc.postprocess([heatmap])
        assert result.shape == (17, 3)

    @pytest.mark.pybinding
    def test_argmax_locates_peak(self):
        """Keypoint 0 should be at (2, 1) for given heatmap"""
        proc = dx_postprocess.VitPosePostProcess(192, 256)
        heatmap = np.zeros((1, 17, 4, 3), dtype=np.float32)
        heatmap[0, 0, 1, 2] = 1.0  # keypoint 0 peak at row=1, col=2
        result = proc.postprocess([heatmap])
        assert result[0, 0] == pytest.approx(2.0)  # x = col
        assert result[0, 1] == pytest.approx(1.0)  # y = row
        assert result[0, 2] == pytest.approx(1.0)  # conf


@skip_if_no_binding
class TestDOPEPostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.DOPEPostProcess(640, 480)
        assert proc.get_input_width() == 640
        assert proc.get_input_height() == 480

    @pytest.mark.pybinding
    def test_postprocess_shape_9x3(self):
        """Output shape must be [9, 3]"""
        proc = dx_postprocess.DOPEPostProcess(640, 480)
        tensor = np.random.rand(1, 25, 4, 5).astype(np.float32)
        result = proc.postprocess([tensor])
        assert result.shape == (9, 3)

    @pytest.mark.pybinding
    def test_peak_at_known_position(self):
        """Belief channel 2 peak at (3, 1) should be returned"""
        proc = dx_postprocess.DOPEPostProcess(640, 480)
        tensor = np.zeros((1, 25, 4, 5), dtype=np.float32)
        tensor[0, 2, 1, 3] = 0.9  # channel 2, row=1, col=3
        result = proc.postprocess([tensor])
        assert result[2, 0] == pytest.approx(3.0)  # x=col
        assert result[2, 1] == pytest.approx(1.0)  # y=row
        assert result[2, 2] == pytest.approx(0.9)  # conf

    @pytest.mark.pybinding
    def test_requires_9_belief_channels(self):
        """DOPE expects at least 9 belief map channels."""
        proc = dx_postprocess.DOPEPostProcess(640, 480)
        tensor = np.zeros((1, 5, 4, 5), dtype=np.float32)
        with pytest.raises(RuntimeError):
            proc.postprocess([tensor])


@skip_if_no_binding
class TestSuperPointPostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.SuperPointPostProcess(640, 480)
        assert proc.get_input_width() == 640
        assert proc.get_input_height() == 480
        assert proc.get_conf_threshold() == pytest.approx(0.015)
        assert proc.get_top_k() == 500

    @pytest.mark.pybinding
    def test_returns_tuple_of_two_arrays(self):
        """Returns (keypoints [N,3], descriptors [N,256])"""
        proc = dx_postprocess.SuperPointPostProcess(640, 480, conf_threshold=0.0)
        semi = np.ones((1, 65, 6, 8), dtype=np.float32)
        desc = np.ones((1, 256, 6, 8), dtype=np.float32)
        kps, descs = proc.postprocess([semi, desc])
        assert kps.ndim == 2 and kps.shape[1] == 3
        assert descs.ndim == 2 and descs.shape[1] == 256
        assert kps.shape[0] == descs.shape[0]

    @pytest.mark.pybinding
    def test_empty_when_below_threshold(self):
        proc = dx_postprocess.SuperPointPostProcess(640, 480, conf_threshold=0.99)
        semi = np.zeros((1, 65, 4, 4), dtype=np.float32)
        desc = np.zeros((1, 256, 4, 4), dtype=np.float32)
        kps, descs = proc.postprocess([semi, desc])
        assert kps.shape == (0, 3)
        assert descs.shape == (0, 256)

    @pytest.mark.pybinding
    def test_descriptors_l2_normalized(self):
        """Each descriptor must have unit L2 norm"""
        proc = dx_postprocess.SuperPointPostProcess(640, 480, conf_threshold=0.0)
        semi = np.ones((1, 65, 4, 4), dtype=np.float32)
        desc = np.random.rand(1, 256, 4, 4).astype(np.float32) + 0.1
        kps, descs = proc.postprocess([semi, desc])
        if descs.shape[0] > 0:
            norms = np.linalg.norm(descs, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @pytest.mark.pybinding
    def test_returns_heatmap_pixel_coordinates(self):
        proc = dx_postprocess.SuperPointPostProcess(640, 480, conf_threshold=0.5, top_k=1)
        semi = np.zeros((1, 65, 1, 1), dtype=np.float32)
        semi[0, 10, 0, 0] = 5.0  # channel 10 -> (x=2, y=1) in the 8x8 heatmap block
        desc = np.ones((1, 256, 1, 1), dtype=np.float32)
        kps, _ = proc.postprocess([semi, desc])
        expected_score = np.exp(5.0) / (np.exp(5.0) + 63.0)
        assert kps.shape == (1, 3)
        assert kps[0, 0] == pytest.approx(2.0)
        assert kps[0, 1] == pytest.approx(1.0)
        assert kps[0, 2] == pytest.approx(expected_score, rel=1e-5)

    @pytest.mark.pybinding
    def test_dustbin_excluded_from_softmax(self):
        """Setting only dustbin channel high should yield near-zero heatmap everywhere"""
        proc = dx_postprocess.SuperPointPostProcess(640, 480, conf_threshold=0.0)
        semi = np.zeros((1, 65, 2, 2), dtype=np.float32)
        semi[0, 64, :, :] = 100.0  # very high dustbin score
        desc = np.ones((1, 256, 2, 2), dtype=np.float32)
        kps, descs = proc.postprocess([semi, desc])
        assert kps.shape == (256, 3)
        assert descs.shape == (256, 256)
        np.testing.assert_allclose(kps[:, 2], 1.0 / 64.0, atol=1e-6)


@skip_if_no_binding
class TestYOLOPv2PostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.YOLOPv2PostProcess(640, 384)
        assert proc.get_input_width() == 640
        assert proc.get_input_height() == 384

    @pytest.mark.pybinding
    def test_returns_tuple_of_three(self):
        """Returns (detections[M,6], drivable[H,W], lane[H,W])"""
        proc = dx_postprocess.YOLOPv2PostProcess(640, 384)
        det0 = np.zeros((1, 255, 48, 80), dtype=np.float32)
        det1 = np.zeros((1, 255, 24, 40), dtype=np.float32)
        det2 = np.zeros((1, 255, 12, 20), dtype=np.float32)
        driv = np.zeros((1, 2, 48, 80), dtype=np.float32)
        lane = np.zeros((1, 1, 48, 80), dtype=np.float32)
        result = proc.postprocess([det0, det1, det2, driv, lane])
        dets, drivable, lane_mask = result
        assert dets.ndim == 2 and dets.shape[1] == 6
        assert drivable.ndim == 2
        assert lane_mask.ndim == 2

    @pytest.mark.pybinding
    def test_no_detections_when_zero_conf(self):
        proc = dx_postprocess.YOLOPv2PostProcess(640, 384)
        det0 = np.zeros((1, 255, 48, 80), dtype=np.float32)
        det1 = np.zeros((1, 255, 24, 40), dtype=np.float32)
        det2 = np.zeros((1, 255, 12, 20), dtype=np.float32)
        driv = np.zeros((1, 2, 48, 80), dtype=np.float32)
        lane = np.zeros((1, 1, 48, 80), dtype=np.float32)
        dets, _, _ = proc.postprocess([det0, det1, det2, driv, lane])
        assert dets.shape[0] == 0

    @pytest.mark.pybinding
    def test_masks_at_model_output_resolution(self):
        """Masks returned at model output size (fast mode, no resize)"""
        proc = dx_postprocess.YOLOPv2PostProcess(640, 384)
        det0 = np.zeros((1, 255, 48, 80), dtype=np.float32)
        det1 = np.zeros((1, 255, 24, 40), dtype=np.float32)
        det2 = np.zeros((1, 255, 12, 20), dtype=np.float32)
        driv = np.zeros((1, 2, 48, 80), dtype=np.float32)
        lane = np.zeros((1, 1, 48, 80), dtype=np.float32)
        _, drivable, lane_mask = proc.postprocess([det0, det1, det2, driv, lane])
        assert drivable.shape == (48, 80)
        assert lane_mask.shape == (48, 80)

    @pytest.mark.pybinding
    def test_per_class_nms_keeps_different_classes(self):
        """Two overlapping boxes of different classes must BOTH survive NMS"""
        proc = dx_postprocess.YOLOPv2PostProcess(
            640, 384, conf_threshold=0.01, nms_threshold=0.1
        )
        det0 = np.full((1, 255, 48, 80), -10.0, dtype=np.float32)
        det0[0, 4, 0, 0] = 10.0
        det0[0, 5 + 0, 0, 0] = 10.0
        det0[0, 85 + 4, 0, 0] = 10.0
        det0[0, 85 + 5 + 1, 0, 0] = 10.0
        det1 = np.full((1, 255, 24, 40), -10.0, dtype=np.float32)
        det2 = np.full((1, 255, 12, 20), -10.0, dtype=np.float32)
        driv = np.zeros((1, 2, 48, 80), dtype=np.float32)
        lane = np.zeros((1, 1, 48, 80), dtype=np.float32)
        dets, _, _ = proc.postprocess([det0, det1, det2, driv, lane])
        assert dets.shape[0] == 2
        class_ids = {int(dets[i, 5]) for i in range(dets.shape[0])}
        assert class_ids == {0, 1}, "Different classes should both survive per-class NMS"


@skip_if_no_binding
class TestMediaPipeHandPostProcess:
    @pytest.mark.pybinding
    def test_constructor_and_getters(self):
        proc = dx_postprocess.MediaPipeHandPostProcess(192, 0.5, 0.3)
        assert proc.get_input_width() == 192
        assert proc.get_input_height() == 192

    @pytest.mark.pybinding
    def test_returns_nx5_array(self):
        """Returns [K, 5] array of (x1,y1,x2,y2,conf)"""
        proc = dx_postprocess.MediaPipeHandPostProcess(192, 0.5, 0.3)
        reg8 = np.zeros((1, 24, 24, 36), dtype=np.float32)
        cls8 = np.full((1, 24, 24, 2), -10.0, dtype=np.float32)
        reg16 = np.zeros((1, 12, 12, 108), dtype=np.float32)
        cls16 = np.full((1, 12, 12, 6), -10.0, dtype=np.float32)
        result = proc.postprocess([reg8, cls8, reg16, cls16])
        assert result.ndim == 2
        assert result.shape[1] == 5

        flat_reg = np.zeros((2016, 18), dtype=np.float32)
        flat_cls = np.full((2016, 1), -10.0, dtype=np.float32)
        flat_result = proc.postprocess([flat_reg, flat_cls])
        assert flat_result.ndim == 2
        assert flat_result.shape[1] == 5

    @pytest.mark.pybinding
    def test_no_detection_below_threshold(self):
        """All-low scores should produce empty result"""
        proc = dx_postprocess.MediaPipeHandPostProcess(192, 0.5, 0.3)
        reg8 = np.zeros((1, 24, 24, 36), dtype=np.float32)
        cls8 = np.full((1, 24, 24, 2), -10.0, dtype=np.float32)
        reg16 = np.zeros((1, 12, 12, 108), dtype=np.float32)
        cls16 = np.full((1, 12, 12, 6), -10.0, dtype=np.float32)
        result = proc.postprocess([reg8, cls8, reg16, cls16])
        assert result.shape[0] == 0

    @pytest.mark.pybinding
    def test_detection_with_high_score(self):
        """High classifier score should yield at least one detection"""
        proc = dx_postprocess.MediaPipeHandPostProcess(192, 0.3, 0.3)
        reg8 = np.zeros((1, 24, 24, 36), dtype=np.float32)
        cls8 = np.full((1, 24, 24, 2), -10.0, dtype=np.float32)
        cls8[0, 12, 12, 0] = 10.0
        reg16 = np.zeros((1, 12, 12, 108), dtype=np.float32)
        cls16 = np.full((1, 12, 12, 6), -10.0, dtype=np.float32)
        result = proc.postprocess([reg8, cls8, reg16, cls16])
        assert result.shape[0] >= 1
        assert result.shape[1] == 5

        mixed_reg_flat = np.zeros((2016, 18), dtype=np.float32)
        mixed_reg_flat[:, 0] = 192.0
        mixed_cls8 = np.full((1, 24, 24, 2), -10.0, dtype=np.float32)
        mixed_cls8[0, 0, 0, 0] = 10.0
        mixed_result = proc.postprocess([mixed_reg_flat, reg8, mixed_cls8, reg16, cls16])
        assert mixed_result.shape[0] >= 1
        assert mixed_result[0, 0] < 0.2
