# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""PP-Matting decode: a continuous alpha matte, not a class map.

Neither ppmatting .dxnn is published (403), so this is written to PaddleSeg's output
-- a single-channel alpha map in [0,1] at the model input resolution -- and refuses
anything else rather than quietly taking channel 0 of a multi-class map.

Two outputs on purpose: `mask` is the thresholded 0/1 class map, because
SemanticSegmentationVisualizer reads `mask` as class IDs and would otherwise render a
continuous alpha through a class palette; `alpha` carries the real matte for anything
that wants to composite with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import PPMattingPostprocessor  # noqa: E402


def _ctx(orig_w: int = 800, orig_h: int = 600) -> PreprocessContext:
    return PreprocessContext(
        original_width=orig_w, original_height=orig_h,
        input_width=512, input_height=512,
        scale_x=512.0 / orig_w, scale_y=512.0 / orig_h,
        scale=min(512.0 / orig_w, 512.0 / orig_h), pad_x=0, pad_y=0,
    )


def _alpha(values: np.ndarray, layout: str = "nchw") -> np.ndarray:
    if layout == "nchw":
        return values.reshape(1, 1, *values.shape).astype(np.float32)
    if layout == "nhwc":
        return values.reshape(1, *values.shape, 1).astype(np.float32)
    return values.reshape(1, *values.shape).astype(np.float32)


def test_alpha_is_resized_to_the_original_and_kept_continuous():
    values = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    result = PPMattingPostprocessor(512, 512).process([_alpha(values)], _ctx())[0]
    assert result.alpha.shape == (600, 800)
    assert result.alpha.dtype == np.float32
    assert 0.0 <= result.alpha.min() and result.alpha.max() <= 1.0
    # continuous, not two-valued
    assert len(np.unique(np.round(result.alpha, 2))) > 10


def test_mask_is_the_thresholded_class_map_the_visualizer_expects():
    values = np.zeros((512, 512), dtype=np.float32)
    values[:256, :] = 0.9                      # top half foreground
    result = PPMattingPostprocessor(512, 512, {"alpha_threshold": 0.5}
                                    ).process([_alpha(values)], _ctx())[0]
    assert result.mask.shape == (600, 800)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 1}
    assert result.mask[:290, :].all()
    assert not result.mask[310:, :].any()
    assert result.class_ids == [0, 1]
    assert result.class_names == ["background", "foreground"]
    assert (result.width, result.height) == (800, 600)


def test_a_three_dimensional_alpha_map_is_accepted():
    values = np.full((512, 512), 0.8, dtype=np.float32)
    result = PPMattingPostprocessor(512, 512).process([_alpha(values, "nhw")], _ctx())[0]
    assert result.alpha.shape == (600, 800)


def test_a_channels_last_alpha_map_is_accepted():
    values = np.full((512, 512), 0.8, dtype=np.float32)
    result = PPMattingPostprocessor(512, 512).process([_alpha(values, "nhwc")], _ctx())[0]
    assert result.alpha.shape == (600, 800)


def test_logits_outside_zero_one_are_squashed():
    values = np.full((512, 512), 4.0, dtype=np.float32)       # a logit, not a probability
    result = PPMattingPostprocessor(512, 512).process([_alpha(values)], _ctx())[0]
    assert result.alpha.max() <= 1.0
    assert result.alpha.mean() == pytest.approx(1.0 / (1.0 + np.exp(-4.0)), abs=1e-3)


def test_a_multi_class_map_is_refused_rather_than_silently_sliced():
    """Taking channel 0 of a 19-class Cityscapes head would 'work' and be wrong."""
    multi = np.zeros((1, 19, 512, 512), dtype=np.float32)
    with pytest.raises(ValueError) as exc:
        PPMattingPostprocessor(512, 512).process([multi], _ctx())
    message = str(exc.value)
    assert "(1, 19, 512, 512)" in message
    assert "single" in message.lower()
