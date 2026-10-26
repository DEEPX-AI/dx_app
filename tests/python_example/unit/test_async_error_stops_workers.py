"""An error in AsyncRunner's main loop stops the pipeline workers at once.

``_run_pipeline_once`` stopped the workers only on KeyboardInterrupt. Any
other exception from the display/drain loop went straight to the join, which
waits 5 s for each of the five workers: the error surfaced about 25 s late,
with the workers still running. Hermetic: the workers are stand-ins that run
until the stop event is set; no factory, engine or NPU is involved.
"""

import threading
import time

import pytest

from common.runner.async_runner import AsyncRunner


class _LoopError(RuntimeError):
    pass


def _bare_runner():
    """AsyncRunner instance without touching a factory or the NPU."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._stop_event = threading.Event()
    runner._verbose = False
    return runner


def test_an_error_in_the_main_loop_stops_the_workers_before_it_surfaces():
    runner = _bare_runner()
    exited = []

    def worker(name):
        def run(*_args):
            while not runner._stop_event.is_set():
                time.sleep(0.01)
            exited.append(name)
        return run

    for name in ("_read_worker", "_preprocess_worker", "_wait_worker",
                 "_postprocess_worker", "_render_worker"):
        setattr(runner, name, worker(name))

    def failing_drain(_queues):
        time.sleep(0.05)  # the workers are running
        raise _LoopError("boom")

    runner._drain_display_queue = failing_drain

    started = time.perf_counter()
    with pytest.raises(_LoopError):
        runner._run_pipeline_once(None, display=False, save_enabled=False, run_dir=None)
    elapsed = time.perf_counter() - started

    assert runner._stop_event.is_set()
    assert sorted(exited) == sorted(["_read_worker", "_preprocess_worker", "_wait_worker",
                                     "_postprocess_worker", "_render_worker"]), exited
    assert elapsed < 3.0, f"the error surfaced {elapsed:.1f} s late"
