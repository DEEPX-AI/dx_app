# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""EfficientAD's published anomaly map, from the three networks it actually needs.

MEASURED on DX-RT 3.5.0, the EfficientAD-M triple at 256x256::

    efficientad-m-teacher_256x256      out teacher_features [1, 384, 64, 64]
    efficientad-m-student_256x256      out student_features [1, 768, 64, 64]
    efficientad-m-autoencoder_256x256  out teacher_features [1, 384, 64, 64]

768 = 384 + 384: the student predicts the TEACHER with its first half and the
AUTOENCODER with its second. That is the whole reason a one-network example was
wrong -- the magnitude of any single feature map is not a discrepancy, and a
discrepancy is what EfficientAD scores.

    st = mean((teacher     - student[:384])**2, axis=C)
    ae = mean((autoencoder - student[384:])**2, axis=C)
    anomaly = 0.5*norm(st) + 0.5*norm(ae)

The official implementation normalises with q_st/q_ae quantiles fitted on the
training set. A .dxnn carries no such constants, so normalisation is per-image and
the ABSOLUTE number is not comparable to a published MVTec figure. The STRUCTURE --
which pixels disagree, and by how much relative to the rest of the frame -- is.

Teacher and autoencoder are indistinguishable by shape (both 384), so their order is
the contract, not a guess. These tests pin it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import EfficientADPostprocessor  # noqa: E402

C, G = 384, 64          # channels per branch, feature grid


def _ctx(w: int = 256, h: int = 256) -> PreprocessContext:
    return PreprocessContext(original_width=w, original_height=h,
                             input_width=256, input_height=256, scale=1.0,
                             pad_x=0.0, pad_y=0.0)


def _triple(seed: int = 0):
    """A perfectly-predicting student: both branches match their target exactly."""
    rng = np.random.default_rng(seed)
    teacher = rng.standard_normal((1, C, G, G)).astype(np.float32)
    autoenc = rng.standard_normal((1, C, G, G)).astype(np.float32)
    student = np.concatenate([teacher, autoenc], axis=1)      # (1, 768, G, G)
    return student, teacher, autoenc


def test_a_perfect_student_produces_no_hot_spot():
    student, teacher, autoenc = _triple()
    result = EfficientADPostprocessor(256, 256).process([student, teacher, autoenc],
                                                        _ctx())[0]
    assert float(result.heatmap.max()) == pytest.approx(0.0, abs=1e-6)


def test_a_discrepancy_in_the_student_teacher_branch_peaks_where_it_was_planted():
    student, teacher, autoenc = _triple()
    student[0, :C, 10, 20] += 8.0                     # ST branch only
    result = EfficientADPostprocessor(256, 256).process([student, teacher, autoenc],
                                                        _ctx(G * 4, G * 4))[0]
    peak = np.unravel_index(int(np.argmax(result.heatmap)), result.heatmap.shape)
    assert peak[0] // 4 == 10 and peak[1] // 4 == 20


def test_a_discrepancy_in_the_autoencoder_branch_is_not_dropped():
    """Halving the formula to its ST term alone would still pass the test above."""
    student, teacher, autoenc = _triple()
    student[0, C:, 40, 5] += 8.0                      # AE branch only
    result = EfficientADPostprocessor(256, 256).process([student, teacher, autoenc],
                                                        _ctx(G * 4, G * 4))[0]
    peak = np.unravel_index(int(np.argmax(result.heatmap)), result.heatmap.shape)
    assert peak[0] // 4 == 40 and peak[1] // 4 == 5


def test_the_two_branches_are_weighted_equally():
    """Equal-magnitude discrepancies in each branch must score equally."""
    student, teacher, autoenc = _triple()
    student[0, :C, 10, 20] += 8.0
    student[0, C:, 40, 5] += 8.0
    heatmap = EfficientADPostprocessor(256, 256).process([student, teacher, autoenc],
                                                          _ctx(G, G))[0].heatmap
    assert heatmap[10, 20] == pytest.approx(heatmap[40, 5], rel=0.05)


def test_the_teacher_and_the_autoencoder_are_not_interchangeable():
    """Both are (1,384,H,W); only the declared ORDER says which is which."""
    student, teacher, autoenc = _triple()
    student[0, :C, 10, 20] += 8.0
    right = EfficientADPostprocessor(256, 256).process([student, teacher, autoenc],
                                                        _ctx(G, G))[0].heatmap
    swapped = EfficientADPostprocessor(256, 256).process([student, autoenc, teacher],
                                                          _ctx(G, G))[0].heatmap
    assert not np.allclose(right, swapped)


def test_the_heatmap_is_returned_at_the_original_resolution():
    student, teacher, autoenc = _triple()
    result = EfficientADPostprocessor(256, 256).process([student, teacher, autoenc],
                                                        _ctx(640, 480))[0]
    assert result.heatmap.shape == (480, 640)


def test_a_single_network_is_refused_with_the_shapes_it_got():
    """This is exactly the mistake the one-engine example made."""
    _, teacher, _ = _triple()
    with pytest.raises(ValueError) as exc:
        EfficientADPostprocessor(256, 256).process([teacher], _ctx())
    message = str(exc.value)
    assert "384" in message and "768" in message


def test_a_student_whose_channels_do_not_split_in_two_is_refused():
    rng = np.random.default_rng(1)
    bad = rng.standard_normal((1, 500, G, G)).astype(np.float32)
    teacher = rng.standard_normal((1, C, G, G)).astype(np.float32)
    with pytest.raises(ValueError):
        EfficientADPostprocessor(256, 256).process([bad, teacher, teacher], _ctx())


def test_declared_roles_identify_the_networks_whatever_the_arrival_order():
    """`-m` picks the PRIMARY model, so the autoencoder can arrive first."""
    student, teacher, autoenc = _triple()
    student[0, :C, 10, 20] += 8.0
    canonical = EfficientADPostprocessor(256, 256).process(
        [student, teacher, autoenc], _ctx(G, G))[0].heatmap
    reordered = EfficientADPostprocessor(256, 256, {
        "roles": ["autoencoder", "student", "teacher"]
    }).process([autoenc, student, teacher], _ctx(G, G))[0].heatmap
    assert np.allclose(canonical, reordered)


def test_roles_that_do_not_name_every_network_are_refused():
    student, teacher, autoenc = _triple()
    with pytest.raises(ValueError) as exc:
        EfficientADPostprocessor(256, 256, {
            "roles": ["student", "teacher", "teacher"]
        }).process([student, teacher, autoenc], _ctx())
    assert "autoencoder" in str(exc.value)


def test_a_bigger_disagreement_scores_higher_on_another_frame():
    """`score` has to rank FRAMES, so it cannot come from the per-frame normalised map.

    The heatmap is normalised against its own frame -- that is what makes the hot
    region visible whatever the absolute magnitudes -- so its 99th percentile is
    roughly the same for a faint anomaly and a glaring one. A frame-ranking number
    must be read off the raw disagreement instead.
    """
    def frame(magnitude: float):
        student, teacher, autoenc = _triple()
        # Wider than the 99th-percentile tail (1% of 64x64 is ~41 px), so the reported
        # percentile actually lands inside the planted region rather than in the
        # synthetic zero background.
        student[0, :C, 30:40, 30:40] += magnitude
        return EfficientADPostprocessor(256, 256).process(
            [student, teacher, autoenc], _ctx(G, G))[0]

    faint, glaring = frame(1.0), frame(20.0)
    assert glaring.score > faint.score * 2
