"""The preview queue must never throttle the inference pipeline.

`display_queue` is a best-effort sink: a frame that is already stale is worthless,
while blocking on it propagates back-pressure through the render worker into the
inference pipeline and clamps end-to-end FPS to the GUI's rate. These tests pin
the drop-oldest behaviour that replaces the old blocking put.
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "python_example"))

from common.utility.safe_queue import SafeQueue          # noqa: E402
from common.runner.async_runner import _SENTINEL, offer_latest  # noqa: E402


def test_offer_latest_never_blocks_on_a_full_queue():
    q = SafeQueue(maxsize=4)
    dropped = []
    for i in range(200):
        offer_latest(q, i, dropped.append)
    assert q.qsize() <= 4, "queue stayed bounded"
    assert len(dropped) == 196, "every superseded frame was dropped, not queued"


def test_offer_latest_keeps_the_newest_frames():
    q = SafeQueue(maxsize=2)
    for i in range(10):
        offer_latest(q, i)
    drained = []
    while (item := q.try_get()) is not None:
        drained.append(item)
    assert drained == [8, 9], f"newest frames survive, got {drained}"


def test_offer_latest_never_drops_the_sentinel():
    q = SafeQueue(maxsize=2)
    offer_latest(q, "frame-a")
    q.put(_SENTINEL, block=False)
    assert q.full()
    offer_latest(q, "frame-b")        # must not evict the shutdown marker
    remaining = []
    while (item := q.try_get()) is not None:
        remaining.append(item)
    assert _SENTINEL in remaining, "sentinel survived; shutdown cannot deadlock"


def test_slow_consumer_does_not_throttle_producer():
    """The headline property: a 10 ms display must not slow a fast producer."""
    q = SafeQueue(maxsize=4)
    stop = threading.Event()
    produced = [0]

    def producer():
        while not stop.is_set():
            offer_latest(q, produced[0])
            produced[0] += 1

    t = threading.Thread(target=producer)
    t.start()
    consumed = 0
    deadline = time.perf_counter() + 0.5
    while time.perf_counter() < deadline:
        if q.try_get() is not None:
            consumed += 1
        time.sleep(0.01)             # deliberately slow display
    stop.set()
    t.join()

    assert consumed < 80, f"consumer stayed slow ({consumed})"
    assert produced[0] > 20_000, (
        f"producer ran freely despite the slow display (only {produced[0]})")
