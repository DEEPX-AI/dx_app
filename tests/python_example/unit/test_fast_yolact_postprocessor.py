"""Unit tests for the fast YOLACT instance-segmentation postprocessor.

The fast variant keeps YOLACT's full detection / Fast-NMS pipeline and only
changes how prototype masks are turned into per-instance masks: instead of
upsampling every mask to the model input resolution and then resizing again to
the original image, it crops each mask to its bbox at *prototype* resolution and
resizes that small crop directly to the original image (one resize from a small
source). This is the same path-B approximation already used by
``FastInstanceSegPostprocessor`` for YOLOv8-seg, so the final binary masks must
agree with the standard path at high IoU.
"""

import sys
from pathlib import Path

import numpy as np


_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base import PreprocessContext
from common.processors.yolact_postprocessor import YOLACTPostprocessor
from common.processors.fast_yolact_postprocessor import FastYOLACTPostprocessor


_INPUT = 64
_PROTO = 32
_NUM_MASKS = 4


def _config():
    return {"num_masks": _NUM_MASKS}


def _proto_masks():
    """Two clear sigmoid-style blobs at prototype resolution, [K, proto, proto]."""
    masks = np.zeros((2, _PROTO, _PROTO), dtype=np.float32)
    masks[0, 8:24, 8:24] = 1.0          # central blob -> input [16,16,48,48]
    masks[1, 4:12, 18:28] = 1.0         # upper-right blob -> input [36,8,56,24]
    return masks


def _boxes():
    """Input-space pixel boxes matching the proto blobs (proto->input ratio 0.5)."""
    return np.array([[16, 16, 48, 48],
                     [36, 8, 56, 24]], dtype=np.float32)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def _final_masks(proc, masks_proto, boxes, ctx):
    """Run scale+crop then to-original for each instance, return binary masks."""
    scaled = proc._scale_masks_and_crop(masks_proto.copy(), boxes)
    out = []
    for i in range(len(boxes)):
        _, orig = proc._to_original_coords(boxes[i].copy(), scaled[i], ctx)
        out.append((orig > 0.5).astype(np.uint8))
    return scaled, out


def test_fast_yolact_is_subclass_of_standard():
    assert issubclass(FastYOLACTPostprocessor, YOLACTPostprocessor)


def test_fast_yolact_scale_masks_stay_at_prototype_resolution():
    """Standard upsamples to input res; fast keeps prototype res (the whole point)."""
    masks, boxes = _proto_masks(), _boxes()
    std = YOLACTPostprocessor(_INPUT, _INPUT, _config())
    fast = FastYOLACTPostprocessor(_INPUT, _INPUT, _config())

    std_scaled = std._scale_masks_and_crop(masks.copy(), boxes)
    fast_scaled = fast._scale_masks_and_crop(masks.copy(), boxes)

    assert std_scaled.shape == (2, _INPUT, _INPUT)
    assert fast_scaled.shape == (2, _PROTO, _PROTO)


def test_fast_yolact_masks_match_standard_at_high_iou_no_padding():
    masks, boxes = _proto_masks(), _boxes()
    ctx = PreprocessContext(pad_x=0, pad_y=0, scale=1.0,
                            original_width=_INPUT, original_height=_INPUT)
    std = YOLACTPostprocessor(_INPUT, _INPUT, _config())
    fast = FastYOLACTPostprocessor(_INPUT, _INPUT, _config())

    _, std_masks = _final_masks(std, masks, boxes, ctx)
    _, fast_masks = _final_masks(fast, masks, boxes, ctx)

    for i in range(len(boxes)):
        assert _iou(std_masks[i], fast_masks[i]) >= 0.9


def test_fast_yolact_masks_match_standard_at_high_iou_with_letterbox():
    masks, boxes = _proto_masks(), _boxes()
    # Letterbox: original 100x80 letterboxed into 64x64 with horizontal padding.
    ctx = PreprocessContext(pad_x=6, pad_y=0, scale=0.64,
                            original_width=100, original_height=80)
    std = YOLACTPostprocessor(_INPUT, _INPUT, _config())
    fast = FastYOLACTPostprocessor(_INPUT, _INPUT, _config())

    _, std_masks = _final_masks(std, masks, boxes, ctx)
    _, fast_masks = _final_masks(fast, masks, boxes, ctx)

    for i in range(len(boxes)):
        assert _iou(std_masks[i], fast_masks[i]) >= 0.9


def test_fast_yolact_box_mapping_identical_to_standard():
    """Box coordinates must be byte-identical: only the mask path changes."""
    masks, boxes = _proto_masks(), _boxes()
    ctx = PreprocessContext(pad_x=6, pad_y=0, scale=0.64,
                            original_width=100, original_height=80)
    std = YOLACTPostprocessor(_INPUT, _INPUT, _config())
    fast = FastYOLACTPostprocessor(_INPUT, _INPUT, _config())

    std_scaled = std._scale_masks_and_crop(masks.copy(), boxes)
    fast_scaled = fast._scale_masks_and_crop(masks.copy(), boxes)
    for i in range(len(boxes)):
        std_box, _ = std._to_original_coords(boxes[i].copy(), std_scaled[i], ctx)
        fast_box, _ = fast._to_original_coords(boxes[i].copy(), fast_scaled[i], ctx)
        assert np.allclose(std_box, fast_box)


def test_factory_create_fast_postprocessor_returns_fast_yolact():
    from common.base import IInstanceSegFactory
    from common.processors import SimpleResizePreprocessor, YOLACTPostprocessor as _YP
    from common.visualizers import InstanceSegVisualizer

    class _YolactFactory(IInstanceSegFactory):
        def __init__(self):
            self.config = {"num_masks": _NUM_MASKS}

        def create_preprocessor(self, w, h):
            return SimpleResizePreprocessor(w, h)

        def create_postprocessor(self, w, h):
            return _YP(w, h, self.config)

        def create_visualizer(self):
            return InstanceSegVisualizer()

        def get_model_name(self):
            return "yolact_test"

        def get_task_type(self):
            return "instance_segmentation"

    fast = _YolactFactory().create_fast_postprocessor(_INPUT, _INPUT)
    assert isinstance(fast, FastYOLACTPostprocessor)
