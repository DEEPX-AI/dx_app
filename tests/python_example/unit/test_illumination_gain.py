# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The gated illumination gain on SimpleResizePreprocessor.

Trimap-free matting needs a subject separable from its background, and one sample
defeats it completely: `sample/img/sample_person_a1.jpg` is a white shirt and bare skin
against a white studio wall (frame mean 209). MEASURED on
ppmatting-hrnet-w48-composition, the face and hands came out as EXACT zeros -- 92.4% of
the head band was 0.0, so the composite showed the checkerboard straight through the
model's head.

A threshold could not fix that, which is why the gain lives on the INPUT:

* the alpha is saturated bimodal -- 85.2% of the frame is exactly 0.0 and only 0.57%
  lies in (0.1, 0.9) -- so moving `alpha_threshold` from 0.5 to 0.3 shifts the
  foreground by 0.08 percentage points, and 0.5 already sits in the empty valley;
* a multiplicative gain on the OUTPUT cannot help either: g * 0 == 0, and at g=50 the
  head band still only reaches 5.0% above 0.5;
* a contrast stretch is strictly worse -- at x1.5 the head band becomes 100% zeros.

Moving the input's channel means onto 110 does work: inside the detector's person box
(conf 0.925) the foreground went 41.6% -> 56.0% and the head band's exact-zero share
92.4% -> 59.4%, with background leakage 0.01% -> 0.03%.

It is GATED because unconditionally it degrades in-distribution subjects: measured,
distinctions/dog 25.30% -> 20.10% and distinctions/face 3.11% -> 1.01%. With
`mean_target_above=195` only person_a1 (mean 209) is touched and the other five
subjects (means 84-182) pass through bit-identically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.processors import SimpleResizePreprocessor  # noqa: E402

CPP_PRE = (PROJECT_ROOT / "src" / "cpp_example" / "common" / "processors"
           / "simple_resize_preprocessor.hpp")


def _frame(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, np.uint8)


def test_off_by_default_is_a_pure_passthrough():
    """499 variants share this class; the gain must not touch any of them."""
    img = _frame(230)
    plain, _ = SimpleResizePreprocessor(32, 32).process(img)
    assert plain.dtype == np.uint8
    # a uniform frame resizes to the same uniform value
    assert plain.min() == plain.max() == 230


def test_gain_moves_the_channel_mean_onto_the_target():
    img = _frame(220)
    out, _ = SimpleResizePreprocessor(32, 32, mean_target=110).process(img)
    assert out.dtype == np.uint8, "the engine for these models takes uint8"
    assert abs(float(out.mean()) - 110.0) <= 1.0


def test_the_gain_is_per_channel_so_a_colour_cast_is_corrected():
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 240, 200, 160   # strong cast
    out, _ = SimpleResizePreprocessor(32, 32, mean_target=110).process(img)
    # SimpleResizePreprocessor converts BGR->RGB by default, so compare the set
    means = sorted(round(float(out[..., c].mean())) for c in range(3))
    assert means == [110, 110, 110], means


def test_the_gate_skips_a_frame_that_is_not_over_bright():
    """A subject inside the model's brightness range must pass through untouched."""
    img = _frame(130)
    gated = SimpleResizePreprocessor(32, 32, mean_target=110, mean_target_above=195)
    plain = SimpleResizePreprocessor(32, 32)
    a, _ = gated.process(img)
    b, _ = plain.process(img)
    assert np.array_equal(a, b), "the gate let a 130-mean frame through the gain"


def test_the_gate_fires_on_an_over_bright_frame():
    img = _frame(230)
    gated = SimpleResizePreprocessor(32, 32, mean_target=110, mean_target_above=195)
    plain = SimpleResizePreprocessor(32, 32)
    a, _ = gated.process(img)
    b, _ = plain.process(img)
    assert not np.array_equal(a, b)
    assert abs(float(a.mean()) - 110.0) <= 1.0


@pytest.mark.parametrize("value", [84, 107, 129, 130, 182, 195])
def test_every_measured_in_distribution_mean_is_below_the_gate(value):
    """The five subjects the ungated gain degraded must all be left alone.

    Their frame means were 84, 107, 129, 130 and 182; 195 is the boundary itself.
    """
    img = _frame(value)
    gated = SimpleResizePreprocessor(32, 32, mean_target=110, mean_target_above=195)
    plain = SimpleResizePreprocessor(32, 32)
    assert np.array_equal(gated.process(img)[0], plain.process(img)[0])


def test_rounding_matches_opencv_saturate_cast():
    """astype() truncates, cv::saturate_cast rounds -- they differ by 1 on half of all
    values, which flipped matte decisions along every edge and showed up as a
    cross-tree foreground difference (14.26% python vs 14.34% cpp)."""
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    img = np.repeat(img[:, :, None], 3, axis=2)
    out, _ = SimpleResizePreprocessor(16, 16, mean_target=110).process(img)
    gain = 110.0 / float(img[..., 0].mean())
    expect = np.clip(np.rint(img.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    # channel 0 of the output is the RGB-converted channel 2 of the input, but the
    # frame is grey so every channel carries the same values
    assert np.array_equal(np.sort(out[..., 0].ravel()),
                          np.sort(expect[..., 0].ravel()))


def test_the_cpp_preprocessor_carries_the_same_two_knobs_and_gate():
    """Both trees must gate identically or the same config produces two results."""
    assert CPP_PRE.is_file(), CPP_PRE
    src = CPP_PRE.read_text(encoding="utf-8")
    assert "float mean_target = 0.f" in src
    assert "float mean_target_above = -1.f" in src
    # same comparison direction as Python: strictly greater than the gate
    assert "overall > mean_target_above_" in src
    assert "mean_target_ > 0.f" in src, "C++ must treat 0 as off, like Python's None"
