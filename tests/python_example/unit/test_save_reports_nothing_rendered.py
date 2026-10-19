# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""``--save`` that writes nothing has to say so, and say why.

A comparison visualizer -- ArcFace, CasViT Re-ID, the CLIP embedding variants -- keeps
the FIRST image as its reference and returns no frame for it. Run against a single
image, the whole pipeline therefore succeeds, prints a performance summary, creates a
run directory, and saves no picture. Nothing in that sequence is wrong, and nothing in
it is explained either: the observed report was that the ArcFace result image was never written
after a run that behaved exactly as designed.

So the silence is the defect. These tests pin the message and its hint.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.runner.sync_runner import SyncRunner  # noqa: E402
from common.visualizers import EmbeddingVisualizer  # noqa: E402


class _Factory:
    def get_task_type(self):
        return "embedding"


def _runner(visualizer) -> SyncRunner:
    runner = SyncRunner(_Factory())
    runner.visualizer = visualizer
    return runner


def test_a_comparison_visualizer_declares_that_it_needs_a_reference():
    """The runner must not have to guess which visualizers work this way."""
    assert getattr(EmbeddingVisualizer, "NEEDS_REFERENCE", False) is True


def test_saving_nothing_explains_the_reference_frame_and_names_a_fix(tmp_path, caplog):
    class _RefVisualizer:
        NEEDS_REFERENCE = True

    runner = _runner(_RefVisualizer())
    with caplog.at_level(logging.WARNING):
        runner._save_image_output(None, "sample/img/sample_face.jpg",
                                  save_enabled=True, run_dir=tmp_path)
    message = caplog.text
    assert "sample_face.jpg" in message
    assert "reference" in message.lower()
    # The hint has to be actionable, not just an explanation.
    assert "face_pair" in message


def test_the_warning_is_not_repeated_for_every_frame(tmp_path, caplog):
    """A 300-frame video must not print 300 identical warnings."""
    class _RefVisualizer:
        NEEDS_REFERENCE = True

    runner = _runner(_RefVisualizer())
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            runner._save_image_output(None, "sample/img/sample_face.jpg",
                                      save_enabled=True, run_dir=tmp_path)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_a_plain_visualizer_that_renders_nothing_is_still_reported(tmp_path, caplog):
    """Without the reference semantics the cause differs, so the hint must not claim it."""
    class _PlainVisualizer:
        pass

    runner = _runner(_PlainVisualizer())
    with caplog.at_level(logging.WARNING):
        runner._save_image_output(None, "sample/img/sample_dog.jpg",
                                  save_enabled=True, run_dir=tmp_path)
    message = caplog.text
    assert "sample_dog.jpg" in message
    assert "face_pair" not in message


def test_a_rendered_frame_is_saved_and_says_nothing(tmp_path, caplog):
    class _PlainVisualizer:
        pass

    runner = _runner(_PlainVisualizer())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        runner._save_image_output(frame, "sample/img/sample_dog.jpg",
                                  save_enabled=True, run_dir=tmp_path)
    assert (tmp_path / "sample_dog_result.jpg").is_file()
    assert caplog.text == ""


def test_nothing_rendered_leaves_no_empty_directory_behind(tmp_path):
    """An empty per-image folder reads exactly like a write that failed.

    The batch path used to mkdir one folder per input up front, so a comparison
    visualizer's reference frame left `1_reference/` sitting empty next to the two
    real outputs -- which is how "the result image was never written" looked from outside.
    """
    class _RefVisualizer:
        NEEDS_REFERENCE = True

    sub_dir = tmp_path / "1_reference"
    runner = _runner(_RefVisualizer())
    runner._save_image_output(None, "sample/img/1_reference.jpg",
                              save_enabled=True, run_dir=sub_dir)
    assert not sub_dir.exists()


def test_a_rendered_frame_creates_its_directory_on_demand(tmp_path):
    class _PlainVisualizer:
        pass

    sub_dir = tmp_path / "2_same"
    runner = _runner(_PlainVisualizer())
    runner._save_image_output(np.zeros((8, 8, 3), dtype=np.uint8),
                              "sample/img/2_same.jpg",
                              save_enabled=True, run_dir=sub_dir)
    assert (sub_dir / "2_same_result.jpg").is_file()
