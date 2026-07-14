"""Unit tests for the generic fast instance-segmentation postprocessor."""

import sys
from pathlib import Path

import numpy as np


_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base import PreprocessContext
from common.processors import FastInstanceSegPostprocessor
from common.processors.instance_seg_postprocessor import InstanceSegPostprocessor


_INPUT = 64
_PROTO = 16
_NUM_CLASSES = 3
_NUM_MASKS = 4


def _config():
    return {
        "score_threshold": 0.3,
        "nms_threshold": 0.45,
        "num_classes": _NUM_CLASSES,
        "num_masks": _NUM_MASKS,
    }


def _ctx():
    return PreprocessContext(pad_x=0, pad_y=0, scale=1.0,
                             original_width=_INPUT, original_height=_INPUT)


def _build_outputs():
    """Build a minimal YOLO-seg style (det, proto) pair with two clear blobs."""
    channels = 4 + _NUM_CLASSES + _NUM_MASKS  # 11
    n = 20
    det = np.zeros((n, channels), dtype=np.float32)

    # Background rows: tiny class scores far away from objects.
    det[:, 4:4 + _NUM_CLASSES] = 0.01
    det[:, 0:4] = (1, 1, 2, 2)

    # Two strong detections with distinct central boxes (cx, cy, w, h).
    det[0, 0:4] = (32, 32, 32, 32)          # box ~[16,16,48,48]
    det[0, 4 + 0] = 0.95                     # class 0
    det[0, 4 + _NUM_CLASSES:] = (4.0, 0, 0, 0)  # mask coef -> proto ch0

    det[1, 0:4] = (48, 16, 16, 16)          # box ~[40,8,56,24]
    det[1, 4 + 1] = 0.90                     # class 1
    det[1, 4 + _NUM_CLASSES:] = (0, 4.0, 0, 0)  # mask coef -> proto ch1

    det_out = det[np.newaxis, ...]           # [1, N, C], shape[1]>shape[2] -> no transpose

    proto = np.full((_NUM_MASKS, _PROTO, _PROTO), -10.0, dtype=np.float32)
    # ch0 positive in central block -> blob for detection 0
    proto[0, 4:12, 4:12] = 10.0
    # ch1 positive in upper-right block -> blob for detection 1
    proto[1, 2:6, 10:14] = 10.0
    proto_out = proto[np.newaxis, ...]       # [1, M, mh, mw]

    return [det_out, proto_out]


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def test_fast_instance_seg_shares_detection_pipeline_with_standard():
    outputs, ctx, cfg = _build_outputs(), _ctx(), _config()
    std = InstanceSegPostprocessor(_INPUT, _INPUT, cfg,
                                   transposed=False, has_objectness=False).process(outputs, ctx)
    fast = FastInstanceSegPostprocessor(_INPUT, _INPUT, cfg,
                                        transposed=False, has_objectness=False).process(outputs, ctx)

    assert len(fast) == len(std) >= 2
    std_keyed = sorted(((d.class_id, tuple(round(v, 3) for v in d.box)) for d in std))
    fast_keyed = sorted(((d.class_id, tuple(round(v, 3) for v in d.box)) for d in fast))
    # Boxes/classes come from the shared NMS pipeline -> identical.
    assert fast_keyed == std_keyed


def test_fast_instance_seg_masks_match_standard_at_high_iou():
    outputs, ctx, cfg = _build_outputs(), _ctx(), _config()
    std = InstanceSegPostprocessor(_INPUT, _INPUT, cfg,
                                   transposed=False, has_objectness=False).process(outputs, ctx)
    fast = FastInstanceSegPostprocessor(_INPUT, _INPUT, cfg,
                                        transposed=False, has_objectness=False).process(outputs, ctx)

    std_by_cls = {d.class_id: d.mask for d in std}
    fast_by_cls = {d.class_id: d.mask for d in fast}
    assert set(fast_by_cls) == set(std_by_cls)
    for cls_id, std_mask in std_by_cls.items():
        fast_mask = fast_by_cls[cls_id]
        assert fast_mask.shape == std_mask.shape == (_INPUT, _INPUT)
        # A+B mixed: nearly identical inside the bbox, boundary-only differences.
        assert _iou(std_mask, fast_mask) >= 0.9


def test_fast_instance_seg_model_name_is_generic():
    assert FastInstanceSegPostprocessor(_INPUT, _INPUT).get_model_name() == "fast_instance_seg"


def test_factory_fast_postprocessor_inherits_create_postprocessor_overrides():
    """Regression: factory-level overrides must reach the fast variant.

    Some factories inject settings directly inside ``create_postprocessor``
    instead of via ``self.config`` (e.g. FastSAM forces ``num_classes=1``,
    ``score_threshold=0.5``, ``nms_threshold=0.65``). ``create_fast_postprocessor``
    must honor the STANDARD instance's resolved config, not the raw
    ``self.config``. Previously it used ``self.config`` (config.json) and dropped
    the overrides, so the fast path decoded / ran NMS differently from standard
    (measured on NPU: FastSAM mask IoU collapsed to ~0.37 with mismatched
    171 vs 189 detections instead of >0.9 with matching 171 vs 171).
    """
    from common.base import IInstanceSegFactory

    class _OverrideFactory(IInstanceSegFactory):
        def __init__(self):
            self.config = {"score_threshold": 0.4, "nms_threshold": 0.45}

        def create_preprocessor(self, input_width, input_height):
            return None

        def create_postprocessor(self, input_width, input_height):
            cfg = {**self.config, "num_classes": 1,
                   "score_threshold": 0.5, "nms_threshold": 0.65}
            return InstanceSegPostprocessor(input_width, input_height, cfg,
                                            transposed=True, has_objectness=False)

        def create_visualizer(self):
            return None

        def get_model_name(self):
            return "override_seg"

        def get_task_type(self):
            return "instance_segmentation"

    fast = _OverrideFactory().create_fast_postprocessor(_INPUT, _INPUT)
    assert isinstance(fast, FastInstanceSegPostprocessor)
    assert (fast.num_classes, fast.score_threshold, fast.nms_threshold) == (1, 0.5, 0.65)

