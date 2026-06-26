"""Unit tests for the fast YOLOv5/YOLOv7 raw-decode postprocessor.

The fast variant keeps the standard ``YOLOv5Postprocessor`` pipeline byte-for-byte
and only changes *where* the per-anchor class sigmoid is computed: instead of
running ``sigmoid`` over the full class grid (``[A, num_classes, H, W]``) for every
anchor and then discarding ~99% of them by the objectness threshold, it first
gates anchors by objectness (a monotonic transform, so the gate is identical) and
computes the class sigmoid only on the survivors.

Because objectness gating uses the *same* ``obj >= obj_threshold`` comparison on the
*same* sigmoid values, and box/class decode for survivors uses the *same* formulas,
the produced detections must be identical to the standard path -- this is an exact
optimization, not a path-B approximation. Every test therefore asserts equality,
not merely high overlap.
"""

import sys
from pathlib import Path

import numpy as np


_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base import PreprocessContext
from common.processors.yolo_postprocessor import YOLOv5Postprocessor
from common.processors.fast_yolo_postprocessor import FastYOLOv5Postprocessor


_INPUT = 320  # strides 8/16/32 -> grids 40/20/10
_C = 85       # 5 + 80 classes


def _ctx():
    return PreprocessContext(pad_x=0, pad_y=0, scale=1.0,
                             original_width=_INPUT, original_height=_INPUT)


def _grids():
    return [(40, 40), (20, 20), (10, 10)]


def _raw_5d_outputs(hot):
    """Build 3 raw [1, A=3, H, W, C=85] tensors with a few hot anchors.

    ``hot`` is a list of (scale_idx, y, x, cls) tuples that get a high objectness
    and class logit so they survive the obj/conf thresholds.
    """
    outs = []
    for (h, w) in _grids():
        outs.append(np.full((1, 3, h, w, _C), -10.0, dtype=np.float32))
    for (si, y, x, cls) in hot:
        outs[si][0, 0, y, x, 4] = 8.0          # objectness logit
        outs[si][0, 0, y, x, 5 + cls] = 8.0     # class logit
        outs[si][0, 0, y, x, 0:4] = 0.0         # tx/ty/tw/th -> sigmoid 0.5
    return outs


def _raw_4d_nchw_outputs(hot):
    """Build 3 raw [1, C=255, H, W] NCHW tensors (channel = a*85 + field)."""
    outs = []
    for (h, w) in _grids():
        outs.append(np.full((1, 3 * _C, h, w), -10.0, dtype=np.float32))
    for (si, y, x, cls) in hot:
        outs[si][0, 0 * _C + 4, y, x] = 8.0
        outs[si][0, 0 * _C + 5 + cls, y, x] = 8.0
        outs[si][0, 0 * _C + 0:0 * _C + 4, y, x] = 0.0
    return outs


def _raw_4d_nhwc_outputs(hot):
    """Build 3 raw [1, H, W, C=255] NHWC tensors."""
    nchw = _raw_4d_nchw_outputs(hot)
    return [np.ascontiguousarray(o.transpose(0, 2, 3, 1)) for o in nchw]


def _same(a, b, *, atol=1e-5):
    """Assert two DetectionResult lists are element-wise identical."""
    assert len(a) == len(b), f"count differs: {len(a)} vs {len(b)}"
    for da, db in zip(a, b):
        assert da.class_id == db.class_id, f"class_id {da.class_id} vs {db.class_id}"
        assert abs(da.confidence - db.confidence) <= atol, \
            f"conf {da.confidence} vs {db.confidence}"
        assert np.allclose(da.box, db.box, atol=atol), f"box {da.box} vs {db.box}"


# --------------------------------------------------------------------------


def test_fast_yolo_is_subclass_of_standard():
    assert issubclass(FastYOLOv5Postprocessor, YOLOv5Postprocessor)


def test_fast_yolo_raw_5d_parity():
    hot = [(0, 5, 5, 0), (1, 3, 3, 1), (2, 2, 2, 2)]
    outs = _raw_5d_outputs(hot)
    std = YOLOv5Postprocessor(_INPUT, _INPUT, {})
    fast = FastYOLOv5Postprocessor(_INPUT, _INPUT, {})
    res_std = std.process([o.copy() for o in outs], _ctx())
    res_fast = fast.process([o.copy() for o in outs], _ctx())
    assert len(res_std) >= 1
    _same(res_std, res_fast)


def test_fast_yolo_multiscale_4d_nchw_parity():
    hot = [(0, 7, 7, 3), (1, 4, 4, 4), (2, 1, 1, 5)]
    outs = _raw_4d_nchw_outputs(hot)
    std = YOLOv5Postprocessor(_INPUT, _INPUT, {})
    fast = FastYOLOv5Postprocessor(_INPUT, _INPUT, {})
    res_std = std.process([o.copy() for o in outs], _ctx())
    res_fast = fast.process([o.copy() for o in outs], _ctx())
    assert len(res_std) >= 1
    _same(res_std, res_fast)


def test_fast_yolo_multiscale_4d_nhwc_parity():
    hot = [(0, 6, 9, 0), (1, 2, 8, 7), (2, 3, 3, 9)]
    outs = _raw_4d_nhwc_outputs(hot)
    std = YOLOv5Postprocessor(_INPUT, _INPUT, {})
    fast = FastYOLOv5Postprocessor(_INPUT, _INPUT, {})
    res_std = std.process([o.copy() for o in outs], _ctx())
    res_fast = fast.process([o.copy() for o in outs], _ctx())
    assert len(res_std) >= 1
    _same(res_std, res_fast)


def test_fast_yolo_decoded_single_tensor_identical():
    """Already-decoded [1, N, 85] output: fast must be a pure no-op vs standard."""
    rng = np.random.default_rng(0)
    n = 200
    out = np.zeros((1, n, _C), dtype=np.float32)
    out[0, :, 0:4] = rng.uniform(10, 300, size=(n, 4))      # cx,cy,w,h pixels
    out[0, :, 2:4] = rng.uniform(10, 60, size=(n, 2))        # keep w,h small
    out[0, :, 4] = rng.uniform(0, 1, size=n)                 # objectness
    out[0, :, 5:] = rng.uniform(0, 1, size=(n, 80))          # class probs
    out[0, 0, 4] = 0.9; out[0, 0, 5] = 0.9
    out[0, 50, 4] = 0.8; out[0, 50, 10] = 0.85
    std = YOLOv5Postprocessor(_INPUT, _INPUT, {})
    fast = FastYOLOv5Postprocessor(_INPUT, _INPUT, {})
    res_std = std.process([out.copy()], _ctx())
    res_fast = fast.process([out.copy()], _ctx())
    _same(res_std, res_fast)


def test_fast_yolo_no_detections_returns_empty():
    outs = _raw_5d_outputs([])  # every objectness logit is -10 -> sigmoid << 0.25
    std = YOLOv5Postprocessor(_INPUT, _INPUT, {})
    fast = FastYOLOv5Postprocessor(_INPUT, _INPUT, {})
    assert std.process([o.copy() for o in outs], _ctx()) == []
    assert fast.process([o.copy() for o in outs], _ctx()) == []


def test_fast_yolo_factory_autodetect():
    from common.base import IDetectionFactory
    from common.processors import SimpleResizePreprocessor, YOLOv5Postprocessor as _YP
    from common.visualizers import DetectionVisualizer

    class _DetFactory(IDetectionFactory):
        def __init__(self):
            self.config = {}

        def create_preprocessor(self, w, h):
            return SimpleResizePreprocessor(w, h)

        def create_postprocessor(self, w, h):
            return _YP(w, h, self.config)

        def create_visualizer(self):
            return DetectionVisualizer()

        def get_model_name(self):
            return "yolov7_test"

        def get_task_type(self):
            return "object_detection"

    fast = _DetFactory().create_fast_postprocessor(_INPUT, _INPUT)
    assert isinstance(fast, FastYOLOv5Postprocessor)
