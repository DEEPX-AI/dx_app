"""Unit tests for the AsyncRunner headless save path.

Two regressions are covered:

1. ``_setup_video_writer`` probed a non-existent ``input_source._source``
   attribute, so ``str(input_source)`` ("<VideoSource object at 0x...>") was
   handed to ``cv2.VideoCapture``. The probe yielded 0x0, no writer was created
   and ``--save`` produced no video at all.
2. ``_push_sentinel`` evicted a queued item to make room for the sentinel, so
   end-of-stream dropped real frames (300 in -> 297 processed).
"""

import threading

import pytest

from common.runner.async_runner import _SENTINEL, AsyncRunner
from common.utility.safe_queue import SafeQueue


class _StubVideoSource:
    """Mimics the IInputSource accessors used by the writer setup."""

    def __init__(self, width=1920, height=1080, fps=30.0, path="clip.mp4"):
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_fps(self):
        return self._fps


def _bare_runner():
    """AsyncRunner instance without touching a factory or the NPU."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._video_writer = None
    runner._stop_event = threading.Event()
    return runner


class TestVideoWriterSetup:
    def test_writer_created_from_input_source_accessors(self, tmp_path):
        runner = _bare_runner()
        runner._setup_video_writer(_StubVideoSource(), True, tmp_path, True)
        assert runner._video_writer is not None, \
            "--save must create a VideoWriter for a video input source"
        runner._video_writer.release()
        assert list(tmp_path.glob("output.*")), "no output video file was created"

    def test_no_writer_when_save_disabled(self, tmp_path):
        runner = _bare_runner()
        runner._setup_video_writer(_StubVideoSource(), False, tmp_path, True)
        assert runner._video_writer is None

    def test_no_writer_for_image_input(self, tmp_path):
        runner = _bare_runner()
        runner._setup_video_writer(_StubVideoSource(), True, tmp_path, False)
        assert runner._video_writer is None

    def test_bad_geometry_is_reported_not_swallowed(self, tmp_path, caplog):
        runner = _bare_runner()
        with caplog.at_level("WARNING"):
            runner._setup_video_writer(_StubVideoSource(width=0, height=0),
                                       True, tmp_path, True)
        assert runner._video_writer is None
        assert any("video" in r.message.lower() for r in caplog.records), \
            "an undeterminable input geometry must be logged, not swallowed"


class TestSentinelDoesNotDropFrames:
    def test_sentinel_waits_for_room_instead_of_evicting(self):
        runner = _bare_runner()
        q = SafeQueue(maxsize=2)
        assert q.put("frame-1")
        assert q.put("frame-2")

        pusher = threading.Thread(target=runner._push_sentinel, args=(q,))
        pusher.start()
        try:
            # Consumer makes room only after the sentinel push is already blocked.
            assert q.get(timeout=1.0) == "frame-1"
            assert q.get(timeout=1.0) == "frame-2"
            assert q.get(timeout=1.0) is _SENTINEL
        finally:
            pusher.join(timeout=5.0)
        assert not pusher.is_alive()

    def test_sentinel_still_lands_when_nothing_consumes(self):
        """Shutdown must never hang, even if the consumer is already gone."""
        runner = _bare_runner()
        q = SafeQueue(maxsize=1)
        assert q.put("frame-1")
        runner._stop_event.set()

        done = threading.Event()

        def _push():
            runner._push_sentinel(q)
            done.set()

        pusher = threading.Thread(target=_push)
        pusher.start()
        pusher.join(timeout=10.0)
        assert done.is_set(), "_push_sentinel must not block forever on shutdown"

        items = []
        while True:
            item = q.try_get()
            if item is None:
                break
            items.append(item)
        assert _SENTINEL in items
