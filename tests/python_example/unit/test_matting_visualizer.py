# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The matting panel: original | alpha matte | composite.

Why a visualizer needed its own tests. ``ppmatting`` was configured with
``SemanticSegmentationVisualizer``, which reads ``SegmentationResult.mask`` as CLASS
IDS and paints it through a 19-colour Cityscapes palette. Given a matting result that
renders the thresholded 0/1 map in two arbitrary palette colours and discards
``alpha`` entirely -- the continuous opacity that is the only thing matting produces.
The output looked like a plausible segmentation, so nothing about it read as broken.

These tests pin that the matte is what gets shown, and that the composite really is
the matting equation rather than a hard cut.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import SegmentationResult  # noqa: E402
from common.visualizers import MattingVisualizer  # noqa: E402
from common.visualizers.matting_visualizer import checkerboard  # noqa: E402

H, W = 40, 60


def _frame(value: int = 200) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def _result(alpha: np.ndarray | None, mask: np.ndarray | None = None):
    return SegmentationResult(
        mask=mask if mask is not None
        else (alpha > 0.5).astype(np.uint8) if alpha is not None else np.array([]),
        alpha=alpha, width=W, height=H,
        class_ids=[0, 1], class_names=["background", "foreground"])


def test_three_panels_side_by_side():
    out = MattingVisualizer().visualize(_frame(), [_result(np.zeros((H, W), np.float32))])
    # three panels plus two 2px dividers
    assert out.shape == (H, W * 3 + 4, 3)


def test_the_middle_panel_is_the_matte_itself_not_a_class_map():
    """A mid alpha must render as mid grey, which a 0/1 class map cannot produce."""
    alpha = np.full((H, W), 0.25, np.float32)
    out = MattingVisualizer().visualize(_frame(), [_result(alpha)])
    matte = out[:, W + 2:2 * W + 2]
    # ignore the caption band at the top
    body = matte[30:, :]
    assert body.min() == body.max() == round(0.25 * 255)
    assert (body[..., 0] == body[..., 1]).all() and (body[..., 1] == body[..., 2]).all()


def test_composite_is_the_matting_equation():
    """``a*F + (1-a)*B`` against the checkerboard, not a thresholded cut."""
    vis = MattingVisualizer()
    frame = _frame(240)
    alpha = np.full((H, W), 0.5, np.float32)
    comp = vis.composite(frame, alpha)
    back = checkerboard(H, W, vis.square)
    expect = (frame.astype(np.float32) * 0.5 + back.astype(np.float32) * 0.5)
    assert np.abs(comp.astype(np.float32) - expect).max() <= 1.0
    # A hard cut would reproduce either the frame or the background exactly.
    assert not np.array_equal(comp, frame)
    assert not np.array_equal(comp, back)


def test_alpha_one_keeps_the_foreground_and_alpha_zero_shows_the_backdrop():
    vis = MattingVisualizer()
    frame = _frame(123)
    assert np.array_equal(vis.composite(frame, np.ones((H, W), np.float32)), frame)
    assert np.array_equal(vis.composite(frame, np.zeros((H, W), np.float32)),
                          checkerboard(H, W, vis.square))


def test_a_matte_at_model_resolution_is_resized_to_the_frame():
    """The postprocessor returns the matte at original size, but not every caller does."""
    alpha = np.zeros((16, 16), np.float32)
    alpha[8:, :] = 1.0
    out = MattingVisualizer().visualize(_frame(), [_result(alpha)])
    assert out.shape == (H, W * 3 + 4, 3)


def test_falls_back_to_the_binary_mask_when_no_alpha_is_present():
    """A hard cut is honest; crashing on a maskless result is not."""
    mask = np.zeros((H, W), np.uint8)
    mask[:, 30:] = 1
    out = MattingVisualizer().visualize(_frame(), [_result(None, mask=mask)])
    assert out.shape == (H, W * 3 + 4, 3)
    matte = out[30:, W + 2:2 * W + 2]
    assert set(np.unique(matte)) <= {0, 255}


def test_no_results_returns_the_frame_untouched():
    frame = _frame()
    out = MattingVisualizer().visualize(frame, [])
    assert np.array_equal(out, frame)
    assert out is not frame


def test_show_panels_false_returns_only_the_composite():
    vis = MattingVisualizer({"show_panels": False})
    out = vis.visualize(_frame(), [_result(np.ones((H, W), np.float32))])
    assert out.shape == (H, W, 3)


def test_checkerboard_alternates_and_is_grey():
    board = checkerboard(32, 32, square=16)
    assert board.shape == (32, 32, 3)
    assert board[0, 0, 0] != board[0, 16, 0]
    assert board[0, 0, 0] == board[16, 16, 0]
    assert (board[..., 0] == board[..., 2]).all()
