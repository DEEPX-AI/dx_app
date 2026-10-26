"""One Ctrl-C ends a looped sync stream run, not just the current loop.

``SyncRunner._stream_inference_once`` catches the KeyboardInterrupt (to release
the capture and print the summary). It used to report
``quit_requested=False``, so ``--loop N`` started the next pass and one Ctrl-C
only ended one of N passes (a video run with ``--loop 100`` kept going for
minutes after SIGINT).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.runner import sync_runner  # noqa: E402


class _Factory:
    def get_task_type(self):
        return "object_detection"


class _InterruptingCapture:
    """Delivers two frames, then the Ctrl-C arrives during the next read."""

    def __init__(self, *_a, **_k):
        self.reads = 0

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0

    def read(self):
        self.reads += 1
        if self.reads > 2:
            raise KeyboardInterrupt
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        pass


def _runner(monkeypatch, loop):
    runner = sync_runner.SyncRunner(_Factory())
    runner._loop = loop
    runner._save = False
    runner._dump_tensors = False
    runner._verbose = False
    monkeypatch.setattr(sync_runner.cv2, "VideoCapture", _InterruptingCapture)
    monkeypatch.setattr(runner, "_is_sr_tiled", lambda: False)
    monkeypatch.setattr(runner, "_process_stream_frame",
                        lambda frame, n, run_dir, label, do_render=True: {"output_frame": frame})
    monkeypatch.setattr(runner, "_accumulate_stream_metrics",
                        lambda metrics, result, t_read: None)
    monkeypatch.setattr(runner, "_save_stream_frame", lambda frame, writer: 0.0)
    monkeypatch.setattr(runner, "_display_stream_frame",
                        lambda frame, display, n: (0.0, False))
    monkeypatch.setattr(sync_runner, "print_sync_performance_summary",
                        lambda *a, **k: None)
    return runner


def test_interrupted_pass_reports_quit(monkeypatch):
    runner = _runner(monkeypatch, loop=1)
    result = runner._stream_inference_once("video.mp4", False, False, None)
    assert result["count"] == 2
    assert result["quit_requested"] is True


def test_one_ctrl_c_stops_all_loops(monkeypatch):
    runner = _runner(monkeypatch, loop=100)
    passes = []

    def run_once(loop_idx, save_enabled):
        passes.append(loop_idx)
        return runner._stream_inference_once("video.mp4", False, save_enabled, None)

    result = runner._run_looped(loop=100, display=False, save=False, run_once=run_once)
    assert passes == [0]
    assert result["processed_loops"] == 1
