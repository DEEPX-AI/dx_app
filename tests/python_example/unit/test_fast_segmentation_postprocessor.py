"""Unit tests for generic fast semantic segmentation postprocessor."""

import sys
from pathlib import Path

import numpy as np


_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.base import PreprocessContext
from common.processors import FastSegmentationPostprocessor


def test_fast_segmentation_postprocessor_argmaxes_low_resolution_logits_then_resizes():
    logits = np.zeros((1, 3, 2, 3), dtype=np.float32)
    logits[0, 0, :, :] = 0.1
    logits[0, 1, 0, :] = 2.0
    logits[0, 2, 1, :] = 3.0
    ctx = PreprocessContext(original_width=6, original_height=4)

    result = FastSegmentationPostprocessor(num_classes=3).process([logits], ctx)[0]

    assert result.width == 6
    assert result.height == 4
    assert result.mask.shape == (4, 6)
    assert set(result.class_ids) == {1, 2}
    assert np.all(result.mask[:2, :] == 1)
    assert np.all(result.mask[2:, :] == 2)


def test_fast_segmentation_postprocessor_scales_letterbox_crop_to_output_resolution():
    class_map = np.repeat(np.arange(4, dtype=np.int32)[:, None], 4, axis=1)
    ctx = PreprocessContext(
        original_width=8,
        original_height=4,
        input_width=8,
        input_height=8,
        scale=1.0,
        pad_y=2,
    )

    result = FastSegmentationPostprocessor(input_width=8, input_height=8).process([class_map], ctx)[0]

    assert result.mask.shape == (4, 8)
    assert np.all(result.mask[:2, :] == 1)
    assert np.all(result.mask[2:, :] == 2)


def test_fast_segmentation_postprocessor_model_name_is_generic():
    assert FastSegmentationPostprocessor().get_model_name() == "fast_segmentation"
