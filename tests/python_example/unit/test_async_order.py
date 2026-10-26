"""AsyncRunner renders frames in input order even when the engine completes them out of order (U-03)."""
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.runner import async_runner  # noqa: E402


class _Factory:
    def get_task_type(self):
        return "object_detection"


class _OutOfOrderEngine:
    """run_async returns at once; each job finishes after a random delay on its own thread."""

    def __init__(self):
        self._next = 0
        self._done = {}
        self._cv = threading.Condition()
        self._rng = random.Random(7)

    def run_async(self, inputs):
        with self._cv:
            req = self._next
            self._next += 1
            delay = self._rng.uniform(0.0, 0.02)
        frame_id = int(inputs[0].flat[0])

        def finish():
            time.sleep(delay)
            with self._cv:
                self._done[req] = [np.array([frame_id])]
                self._cv.notify_all()

        threading.Thread(target=finish, daemon=True).start()
        return req

    def wait(self, req):
        with self._cv:
            self._cv.wait_for(lambda: req in self._done)
            return self._done.pop(req)


class _Pre:
    def process(self, frame):
        return frame.copy(), None


class _Post:
    def process(self, outputs, ctx):
        return [int(outputs[0][0])]


class _Vis:
    def __init__(self):
        self.seen = []

    def visualize(self, frame, results):
        self.seen.append(results[0])
        return frame


def test_async_runner_renders_frames_in_input_order(tmp_path, monkeypatch):
    # The render worker skips frames that are neither shown nor saved (8d0b748);
    # DXAPP_SAVE_IMAGE makes it render, and so visualize, every frame.
    monkeypatch.setenv("DXAPP_SAVE_IMAGE", str(tmp_path / "frame.png"))
    runner = async_runner.AsyncRunner(_Factory())
    runner.ie = _OutOfOrderEngine()
    runner.preprocessor = _Pre()
    runner.postprocessor = _Post()
    runner.visualizer = _Vis()
    runner._input_dtype = None
    runner._nchw = False
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(60)]
    result = runner._run_pipeline_once(iter(frames), display=False, save_enabled=False,
                                       run_dir=None, is_video=True)
    assert result["count"] == 60
    assert runner.visualizer.seen == list(range(60))
