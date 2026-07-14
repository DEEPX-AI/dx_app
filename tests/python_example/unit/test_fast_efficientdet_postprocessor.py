"""Unit tests for the fast EfficientDet multi-output postprocessor.

The fast variant keeps the standard ``EfficientDetPostprocessor`` decode/NMS
pipeline byte-for-byte and only removes per-frame waste in the BiFPN multi-output
path (``_process_multi_output``):

* the per-anchor ``argmax`` over all classes is computed only for anchors that
  pass the score threshold (the rest are masked out downstream anyway), and
* the regression-vs-absolute coordinate decision (``np.percentile`` / ``np.median``
  over all ~76k anchors) is computed once and cached, since a given model always
  emits the same tensor format.

The fundamental per-anchor ``max`` and the downstream ``_decode_anchor_results`` /
``_decode_results`` are reused unchanged, so the detections must be identical to
the standard path. Tests therefore assert equality, not mere overlap.
"""

import sys
from pathlib import Path

import numpy as np


_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base import PreprocessContext
from common.processors.efficientdet_postprocessor import EfficientDetPostprocessor
from common.processors.fast_efficientdet_postprocessor import FastEfficientDetPostprocessor


_INPUT = 128  # small so anchor generation (P3-P7) stays fast in tests
_NUM_CLASSES = 90  # 1 background + 89 foreground (has_background=True default)


def _ctx():
    return PreprocessContext(pad_x=0, pad_y=0, scale=1.0,
                             original_width=_INPUT, original_height=_INPUT)


def _bifpn_features():
    """Five squeeze-to-ndim3 feature tensors that the decode loop ignores.

    Spatial dims are kept >= 2 so none collapse to ndim<=2 under ``np.squeeze``
    (a 1x1 feature would squeeze to 1D and be mistaken for a score tensor).
    """
    return [np.zeros((1, 88, s, s), dtype=np.float32) for s in (16, 8, 4, 3, 2)]


def _regression_outputs(n, hot):
    """BiFPN-style outputs whose box tensor looks like anchor regressions.

    ``hot`` is a list of (anchor_idx, fg_cls) tuples set above the score
    threshold. Box regressions are small (centered at 0) so the standard path
    takes the anchor-decode branch.
    """
    rng = np.random.default_rng(7)
    box = rng.uniform(-0.4, 0.4, size=(1, n, 4)).astype(np.float32)
    scores = rng.uniform(0.0, 0.1, size=(1, n, _NUM_CLASSES)).astype(np.float32)
    for (ai, cls) in hot:
        scores[0, ai, 1 + cls] = 0.9
        box[0, ai, :] = [0.05, -0.05, 0.1, 0.1]
    return _bifpn_features() + [box, scores]


def _absolute_outputs(n, hot):
    """Box tensor in absolute pixel coords so the non-regression branch runs."""
    rng = np.random.default_rng(11)
    box = np.zeros((1, n, 4), dtype=np.float32)
    scores = rng.uniform(0.0, 0.1, size=(1, n, _NUM_CLASSES)).astype(np.float32)
    for k, (ai, cls) in enumerate(hot):
        scores[0, ai, 1 + cls] = 0.9
        x1 = 5 + k * 15
        y1 = 8 + k * 10
        box[0, ai, :] = [x1, y1, x1 + 20, y1 + 18]
    return _bifpn_features() + [box, scores]


def _same(a, b, *, atol=1e-5):
    assert len(a) == len(b), f"count differs: {len(a)} vs {len(b)}"
    for da, db in zip(a, b):
        assert da.class_id == db.class_id, f"class_id {da.class_id} vs {db.class_id}"
        assert abs(da.confidence - db.confidence) <= atol, \
            f"conf {da.confidence} vs {db.confidence}"
        assert np.allclose(da.box, db.box, atol=atol), f"box {da.box} vs {db.box}"


# --------------------------------------------------------------------------


def test_fast_efficientdet_is_subclass_of_standard():
    assert issubclass(FastEfficientDetPostprocessor, EfficientDetPostprocessor)


def test_fast_efficientdet_regression_parity():
    outs = _regression_outputs(2000, hot=[(10, 3), (500, 7), (1500, 20)])
    std = EfficientDetPostprocessor(_INPUT, _INPUT, {})
    fast = FastEfficientDetPostprocessor(_INPUT, _INPUT, {})
    res_std = std.process([o.copy() for o in outs], _ctx())
    res_fast = fast.process([o.copy() for o in outs], _ctx())
    assert len(res_std) >= 1
    _same(res_std, res_fast)


def test_fast_efficientdet_absolute_parity():
    outs = _absolute_outputs(2000, hot=[(3, 1), (40, 5), (900, 12)])
    std = EfficientDetPostprocessor(_INPUT, _INPUT, {})
    fast = FastEfficientDetPostprocessor(_INPUT, _INPUT, {})
    res_std = std.process([o.copy() for o in outs], _ctx())
    res_fast = fast.process([o.copy() for o in outs], _ctx())
    assert len(res_std) >= 1
    _same(res_std, res_fast)


def test_fast_efficientdet_no_detections_returns_empty():
    outs = _regression_outputs(1000, hot=[])  # all scores < threshold
    std = EfficientDetPostprocessor(_INPUT, _INPUT, {})
    fast = FastEfficientDetPostprocessor(_INPUT, _INPUT, {})
    assert std.process([o.copy() for o in outs], _ctx()) == []
    assert fast.process([o.copy() for o in outs], _ctx()) == []


def test_fast_efficientdet_caches_coord_decision_and_stays_parity():
    """Second frame reuses the cached coord-format decision yet stays identical."""
    outs = _regression_outputs(2000, hot=[(10, 3), (500, 7)])
    std = EfficientDetPostprocessor(_INPUT, _INPUT, {})
    fast = FastEfficientDetPostprocessor(_INPUT, _INPUT, {})
    assert fast._coord_is_regression is None
    r1_std = std.process([o.copy() for o in outs], _ctx())
    r1_fast = fast.process([o.copy() for o in outs], _ctx())
    assert fast._coord_is_regression is True
    r2_fast = fast.process([o.copy() for o in outs], _ctx())
    _same(r1_std, r1_fast)
    _same(r1_std, r2_fast)


def test_fast_efficientdet_factory_autodetect():
    from common.base import IDetectionFactory
    from common.processors import SimpleResizePreprocessor, EfficientDetPostprocessor as _EP
    from common.visualizers import DetectionVisualizer

    class _EDFactory(IDetectionFactory):
        def __init__(self):
            self.config = {}

        def create_preprocessor(self, w, h):
            return SimpleResizePreprocessor(w, h)

        def create_postprocessor(self, w, h):
            return _EP(w, h, self.config)

        def create_visualizer(self):
            return DetectionVisualizer()

        def get_model_name(self):
            return "efficientdetd1_test"

        def get_task_type(self):
            return "object_detection"

    fast = _EDFactory().create_fast_postprocessor(_INPUT, _INPUT)
    assert isinstance(fast, FastEfficientDetPostprocessor)
