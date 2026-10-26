# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Anomaly feature extraction -- deliberately a feature response, not a score.

EfficientAD's anomaly map needs all three models at once:

    0.5*norm(mean((teacher - student[:384])^2)) + 0.5*norm(mean((ae - student[384:])^2))

and PatchCore's needs a memory bank of training-set features. Each .dxnn here is one
model, and dx_app has no in-tree multi-model app pattern, so each variant ships a
single-model feature-extraction example with a heatmap. That is a real working app; it
is not the EfficientAD score, and `AnomalyResult.score` says so in its docstring.

None of the four .dxnn files is published (403), so the decode is written to the shape
these architectures emit -- a (1, C, H, W) feature map -- and refuses anything else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import AnomalyFeaturePostprocessor  # noqa: E402


def _ctx(orig_w: int = 512, orig_h: int = 384) -> PreprocessContext:
    return PreprocessContext(
        original_width=orig_w, original_height=orig_h,
        input_width=256, input_height=256,
        scale_x=256.0 / orig_w, scale_y=256.0 / orig_h,
        scale=min(256.0 / orig_w, 256.0 / orig_h), pad_x=0, pad_y=0,
    )


def test_reduces_a_feature_map_to_a_heatmap_at_original_resolution():
    features = np.random.default_rng(0).normal(size=(1, 384, 64, 64)).astype(np.float32)
    result = AnomalyFeaturePostprocessor(256, 256).process([features], _ctx())[0]
    assert result.heatmap.shape == (384, 512)
    assert result.heatmap.dtype == np.float32
    assert 0.0 <= result.heatmap.min() and result.heatmap.max() <= 1.0


def test_reports_the_channel_count_so_the_model_is_identifiable():
    """teacher/autoencoder emit 384 channels, student 768 -- worth surfacing."""
    for channels in (384, 768):
        features = np.zeros((1, channels, 64, 64), dtype=np.float32)
        result = AnomalyFeaturePostprocessor(256, 256).process([features], _ctx())[0]
        assert result.channels == channels


def test_a_constant_map_does_not_divide_by_zero():
    features = np.full((1, 384, 64, 64), 7.0, dtype=np.float32)
    result = AnomalyFeaturePostprocessor(256, 256).process([features], _ctx())[0]
    assert np.all(np.isfinite(result.heatmap))
    assert result.heatmap.max() == 0.0, "a featureless map must be uniformly quiet"
    assert np.isfinite(result.score)


def test_the_hottest_region_survives_normalisation_and_resize():
    features = np.zeros((1, 384, 64, 64), dtype=np.float32)
    features[0, :, 8:16, 40:48] = 5.0
    heatmap = AnomalyFeaturePostprocessor(256, 256).process([features], _ctx())[0].heatmap
    # 64x64 grid -> 512x384 frame: rows 8..16 -> y 48..96, cols 40..48 -> x 320..384
    assert heatmap[60:85, 330:375].mean() > 0.9
    assert heatmap[200:, :100].mean() < 0.1


def test_score_is_a_relative_severity_in_zero_one():
    quiet = np.zeros((1, 384, 64, 64), dtype=np.float32)
    quiet[0, :, 0, 0] = 1.0
    loud = np.zeros((1, 384, 64, 64), dtype=np.float32)
    loud[0, :, :32, :32] = 1.0
    post = AnomalyFeaturePostprocessor(256, 256)
    quiet_score = post.process([quiet], _ctx())[0].score
    loud_score = post.process([loud], _ctx())[0].score
    assert 0.0 <= quiet_score <= 1.0 and 0.0 <= loud_score <= 1.0
    assert loud_score > quiet_score


def test_a_channels_last_feature_map_is_accepted():
    features = np.random.default_rng(1).normal(size=(1, 64, 64, 384)).astype(np.float32)
    result = AnomalyFeaturePostprocessor(256, 256).process([features], _ctx())[0]
    assert result.heatmap.shape == (384, 512)
    assert result.channels == 384


def test_a_non_feature_map_output_raises_with_the_actual_shapes():
    with pytest.raises(ValueError) as exc:
        AnomalyFeaturePostprocessor(256, 256).process(
            [np.zeros((1, 1000), dtype=np.float32)], _ctx())
    assert "(1, 1000)" in str(exc.value)
