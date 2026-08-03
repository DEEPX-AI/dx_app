"""Unit tests for scripts/validate_save_mode.py output checking.

A validator that cannot fail is worthless, so these pin the failure modes it
exists to catch: a missing run directory, a video that decodes to nothing, and
a video that is missing its tail frames.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

validate_save_mode = pytest.importorskip("validate_save_mode")

validate_outputs = validate_save_mode.validate_outputs
select_tasks = validate_save_mode.select_tasks
select_modes = validate_save_mode.select_modes


def _run_dir(tmp_path, with_info=True):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    if with_info:
        (d / "run_info.txt").write_text("model: test\n")
    return d


def _write_video(path, frames, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


class TestVideoValidation:
    def test_all_frames_present_passes(self, tmp_path):
        d = _run_dir(tmp_path)
        _write_video(d / "output.mp4", 10)
        ok, detail, saved = validate_outputs(d, "video", 10, visual=True)
        assert ok, detail
        assert saved == 10

    def test_missing_tail_frames_fails(self, tmp_path):
        d = _run_dir(tmp_path)
        _write_video(d / "output.mp4", 7)
        ok, detail, saved = validate_outputs(d, "video", 10, visual=True)
        assert not ok
        assert "7/10" in detail
        assert saved == 7

    def test_zero_frame_video_fails(self, tmp_path):
        d = _run_dir(tmp_path)
        _write_video(d / "output.mp4", 0)
        ok, detail, _ = validate_outputs(d, "video", 10, visual=True)
        assert not ok
        assert "0 frames" in detail

    def test_no_video_at_all_fails(self, tmp_path):
        d = _run_dir(tmp_path)
        ok, detail, _ = validate_outputs(d, "video", 10, visual=True)
        assert not ok
        assert "no output video" in detail

    def test_missing_run_dir_fails(self, tmp_path):
        d = _run_dir(tmp_path, with_info=False)
        _write_video(d / "output.mp4", 10)
        ok, detail, _ = validate_outputs(d, "video", 10, visual=True)
        assert not ok
        assert "run_info.txt" in detail


class TestImageValidation:
    def test_decodable_image_passes(self, tmp_path):
        d = _run_dir(tmp_path)
        cv2.imwrite(str(d / "out.jpg"), np.zeros((10, 10, 3), np.uint8))
        ok, detail, saved = validate_outputs(d, "image", 0, visual=True)
        assert ok, detail
        assert saved == 1

    def test_empty_image_file_fails(self, tmp_path):
        d = _run_dir(tmp_path)
        (d / "out.jpg").write_bytes(b"")
        ok, detail, _ = validate_outputs(d, "image", 0, visual=True)
        assert not ok

    def test_no_image_fails(self, tmp_path):
        d = _run_dir(tmp_path)
        ok, detail, _ = validate_outputs(d, "image", 0, visual=True)
        assert not ok
        assert "no output image" in detail


class TestNonVisualTasks:
    def test_run_dir_is_enough_for_embedding_like_tasks(self, tmp_path):
        d = _run_dir(tmp_path)
        ok, detail, saved = validate_outputs(d, "image", 0, visual=False)
        assert ok, detail
        assert saved is None


class TestSelection:
    def test_task_selection_by_index_and_name(self):
        assert select_tasks("0") == [0]
        by_name = select_tasks("super resolution")
        assert by_name and all(
            "super" in validate_save_mode.DEMOS[i][validate_save_mode.D_LABEL].lower()
            or "super" in validate_save_mode.DEMOS[i][validate_save_mode.D_PYDIR].lower()
            for i in by_name)

    def test_unknown_task_exits(self):
        with pytest.raises(SystemExit):
            select_tasks("no-such-task")

    def test_unknown_mode_exits(self):
        with pytest.raises(SystemExit):
            select_modes("cpp_turbo")

    def test_mode_subset_keeps_registry_order(self):
        modes = select_modes("py_sync,cpp_sync")
        assert [m[0] for m in modes] == ["cpp_sync", "py_sync"]
