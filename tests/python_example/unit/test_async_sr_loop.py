"""AsyncRunner tiled super-resolution stream honours ``--loop``.

``_run_stream_sr`` read the source once whatever ``--loop`` said, while every
other stream path (``_run_stream``) runs ``for loop_idx in range(self._loop)``,
reopens the source per loop and saves only during loop 0. These tests drive the
SR stream path with a fake N-frame capture and a fake tiler; no NPU, no video
file and no real run directory.
"""

import numpy as np
import pytest

import common.runner.async_runner as ar
from common.runner.async_runner import AsyncRunner

N_FRAMES = 5


class _FakeCapture:
    """cv2.VideoCapture stand-in: every open yields N_FRAMES frames."""

    opens = 0

    def __init__(self, source):
        type(self).opens += 1
        self._left = N_FRAMES
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        if self._left == 0:
            return False, None
        self._left -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeWriter:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """A bare AsyncRunner wired to fakes; returns (runner, record)."""
    _FakeCapture.opens = 0
    rec = {"processed": 0, "written": 0, "run_dirs": 0, "summaries": [],
           "writers": []}

    runner = AsyncRunner.__new__(AsyncRunner)
    runner._loop = 1
    runner._save = False
    runner._save_dir = str(tmp_path)
    runner._dump_tensors = False
    runner._verbose = False
    runner._model_path = "fake.dxnn"
    runner._display_size = (8, 8)
    runner._sr_cache = {"scale_x": 2, "scale_y": 2}
    runner._init_sr_cache = lambda: None
    runner._verify_dump = lambda results, frame: None

    def process(frame):
        rec["processed"] += 1
        out = np.zeros((8, 8, 3), dtype=np.uint8)
        return {"output_frame": out, "sr_output": out, "t_pre": 0.0,
                "t_infer": 0.0, "t_post": 0.0, "t_render": 0.0}

    runner._process_sr_frame = process

    def init_writer(run_dir, w, h, fps):
        writer = _FakeWriter()
        rec["writers"].append(writer)
        runner._video_writer_size = (w, h)  # as the real writer setup does
        return writer

    runner._init_video_writer = init_writer

    def create_run_dir(kind, name, save_dir):
        rec["run_dirs"] += 1
        return tmp_path / f"run{rec['run_dirs']}"

    def write_frame(writer, canvas, size):
        rec["written"] += 1

    def summary(metrics, count, elapsed, render):
        rec["summaries"].append(count)

    monkeypatch.setattr(ar.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(ar, "create_run_dir", create_run_dir)
    monkeypatch.setattr(ar, "write_run_info", lambda *a, **k: None)
    monkeypatch.setattr(ar, "write_video_frame", write_frame)
    monkeypatch.setattr(ar, "print_sync_performance_summary", summary)
    monkeypatch.setattr(ar, "_has_display", lambda: False)
    return runner, rec


def test_loop_two_processes_every_frame_twice_and_saves_loop_zero_only(harness):
    runner, rec = harness
    runner._loop = 2
    runner._save = True

    runner._run_stream_sr("clip.avi", display=False)

    assert rec["processed"] == 2 * N_FRAMES
    assert _FakeCapture.opens == 2, "the source is reopened for each loop"
    assert rec["run_dirs"] == 1, "a run dir is created for the saved loop only"
    assert len(rec["writers"]) == 1 and rec["writers"][0].released
    assert rec["written"] == N_FRAMES, "only loop 0 is saved"
    assert rec["summaries"] == [2 * N_FRAMES], \
        "one summary over all loops, counting every frame"


def test_single_loop_is_unchanged(harness):
    runner, rec = harness

    runner._run_stream_sr("clip.avi", display=False)

    assert rec["processed"] == N_FRAMES
    assert _FakeCapture.opens == 1
    assert rec["summaries"] == [N_FRAMES]


def test_dump_tensors_gets_a_run_dir_every_loop(harness):
    runner, rec = harness
    runner._loop = 3
    runner._dump_tensors = True

    runner._run_stream_sr("clip.avi", display=False)

    assert rec["processed"] == 3 * N_FRAMES
    assert rec["run_dirs"] == 3
    assert rec["writers"] == [], "no --save, so no video writer"


def test_interrupt_ends_all_loops(harness):
    runner, rec = harness
    runner._loop = 3
    real = runner._process_sr_frame

    def interrupt_on_third(frame):
        if rec["processed"] == 2:
            raise KeyboardInterrupt
        return real(frame)

    runner._process_sr_frame = interrupt_on_third

    runner._run_stream_sr("clip.avi", display=False)

    assert _FakeCapture.opens == 1, "an interrupt must not start the next loop"
    assert rec["processed"] == 2
    assert rec["summaries"] == [2]


def test_window_close_ends_all_loops(harness, monkeypatch):
    runner, rec = harness
    runner._loop = 3
    runner._show_output = lambda canvas: None
    monkeypatch.setattr(ar, "_has_display", lambda: True)
    monkeypatch.setattr(ar, "_window_should_close", lambda name: True)
    monkeypatch.setattr(ar.cv2, "destroyAllWindows", lambda: None)

    runner._run_stream_sr("clip.avi", display=True)

    assert _FakeCapture.opens == 1, "closing the window must not start the next loop"
    assert rec["processed"] == 1
