"""Python runners stop like the C++ ones (U-06): SIGTERM and the first Ctrl-C are
graceful, repeats within 200 ms are one request, a later Ctrl-C terminates."""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.runner import interrupts  # noqa: E402

MODULE = _SRC / "common" / "runner" / "interrupts.py"

# The real module in a real process, entered the way the runners enter it: a busy
# loop, then a slow wind-down (a video writer and a summary) once the graceful
# stop arrives. A second KeyboardInterrupt in the wind-down prints "second".
CHILD = r'''
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("interrupts", sys.argv[1])
interrupts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interrupts)
with interrupts.interrupt_scope():
    print("ready", flush=True)
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("graceful", flush=True)
        end = time.monotonic() + float(sys.argv[2])
        try:
            while time.monotonic() < end:
                time.sleep(0.01)
        except KeyboardInterrupt:
            # Only a real SIG_DFL kill ends the wind-down without a trace.
            print("second", flush=True)
        print("summary", flush=True)
'''


def _run_child(tmp_path, signals, wind_down):
    child = tmp_path / "child.py"
    child.write_text(CHILD)
    proc = subprocess.Popen([sys.executable, str(child), str(MODULE), str(wind_down)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        for sig, delay in signals:
            time.sleep(delay)
            os.kill(proc.pid, sig)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    return proc.returncode, out.split()


def test_first_request_is_graceful():
    assert interrupts.classify(signal.SIGTERM, 10.0, None) == ("graceful", 10.0)
    assert interrupts.classify(signal.SIGINT, 10.0, None) == ("graceful", 10.0)


def test_repeats_within_the_window_are_the_same_request():
    assert interrupts.classify(signal.SIGINT, 0.1, 0.0) == ("ignore", 0.0)
    assert interrupts.classify(signal.SIGINT, 0.2, 0.0) == ("ignore", 0.0)  # boundary: same request


def test_later_sigint_terminates_and_sigterm_never_does():
    assert interrupts.classify(signal.SIGINT, 0.3, 0.0) == ("terminate", 0.0)
    assert interrupts.classify(signal.SIGTERM, 99.0, 0.0) == ("ignore", 0.0)


def test_install_off_the_main_thread_is_refused():
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(interrupts.install_signal_handlers()))
    worker.start()
    worker.join()
    assert outcome == [False]


def test_each_scope_restores_the_handlers_and_starts_fresh():
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            with interrupts.interrupt_scope() as installed:
                assert installed
                assert signal.getsignal(signal.SIGTERM) is interrupts._handler
                os.kill(os.getpid(), signal.SIGTERM)  # first request of this scope: graceful
                time.sleep(1.0)  # delivery; the handler raises before this ends
        assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_scope_off_the_main_thread_changes_nothing():
    before = signal.getsignal(signal.SIGTERM)
    outcome = []

    def enter():
        with interrupts.interrupt_scope() as installed:
            outcome.append(installed)

    worker = threading.Thread(target=enter)
    worker.start()
    worker.join()
    assert outcome == [False]
    assert signal.getsignal(signal.SIGTERM) is before


class _ThreadInterruptedOnce:
    """A worker whose first join is hit by the stop request."""

    def __init__(self):
        self.joins = 0

    def join(self, timeout=None):
        self.joins += 1
        if self.joins == 1:
            raise KeyboardInterrupt


def test_a_stop_request_during_the_joins_stops_the_workers_first():
    from common.runner.async_runner import AsyncRunner
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._stop_event = threading.Event()
    worker = _ThreadInterruptedOnce()
    runner._join_workers([worker], {})
    assert runner._stop_event.is_set()
    assert worker.joins == 2


def test_sigterm_is_a_graceful_stop(tmp_path):
    assert _run_child(tmp_path, [(signal.SIGTERM, 0.1)], 0.3) == (0, ["graceful", "summary"])


def test_repeated_sigterm_stays_graceful(tmp_path):
    assert _run_child(tmp_path, [(signal.SIGTERM, 0.1), (signal.SIGTERM, 0.5)], 1.0) == (0, ["graceful", "summary"])


def test_a_burst_of_sigint_is_one_request(tmp_path):
    burst = [(signal.SIGINT, 0.1), (signal.SIGINT, 0.0), (signal.SIGINT, 0.0)]
    assert _run_child(tmp_path, burst, 0.5) == (0, ["graceful", "summary"])


def test_a_later_sigint_ends_a_hung_wind_down(tmp_path):
    started = time.monotonic()
    rc, out = _run_child(tmp_path, [(signal.SIGINT, 0.1), (signal.SIGINT, 0.5)], 5.0)
    assert (rc, out) == (-signal.SIGINT, ["graceful"])
    assert time.monotonic() - started < 4.0


def test_sigterm_then_a_later_sigint_terminates(tmp_path):
    rc, out = _run_child(tmp_path, [(signal.SIGTERM, 0.1), (signal.SIGINT, 0.5)], 5.0)
    assert (rc, out) == (-signal.SIGINT, ["graceful"])


# Spec 4.8: SIGINT/SIGTERM outside a stream loop (setup, image mode) is a clean
# end (rc 0), and the handlers the run replaced are back afterwards.
@pytest.mark.parametrize("runner_type", ["SyncRunner", "AsyncRunner"])
def test_an_interrupt_outside_a_stream_loop_ends_the_run_cleanly(runner_type, monkeypatch):
    from common.runner.async_runner import AsyncRunner
    from common.runner.sync_runner import SyncRunner
    cls = {"SyncRunner": SyncRunner, "AsyncRunner": AsyncRunner}[runner_type]
    runner = cls.__new__(cls)
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    during = []

    def interrupted(args):
        during.append((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_run", interrupted)
    try:
        assert runner.run(object()) is None
    except KeyboardInterrupt:  # would end the pytest session, not just this test
        pytest.fail("the interrupt escaped run()")
    assert during == [(interrupts._handler, interrupts._handler)]
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


class _Writer:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _pipeline_runner(monkeypatch):
    """An AsyncRunner whose five workers only wait for the stop request."""
    from common.runner.async_runner import AsyncRunner
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._stop_event = threading.Event()
    runner._verbose = False
    for name in ("_read_worker", "_preprocess_worker", "_wait_worker",
                 "_postprocess_worker", "_render_worker"):
        monkeypatch.setattr(runner, name, lambda *args: runner._stop_event.wait(5.0))
    return runner


def _run_pipeline(runner):
    try:
        return runner._run_pipeline_once(None, display=False, save_enabled=True,
                                         run_dir=None, is_video=True)
    except KeyboardInterrupt:  # would end the pytest session, not just this test
        pytest.fail("the stop request escaped _run_pipeline_once")


def test_a_stop_request_while_the_writer_opens_still_releases_it(monkeypatch):
    runner = _pipeline_runner(monkeypatch)
    writer = _Writer()

    def open_then_interrupted(*args):
        runner._video_writer = writer
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_setup_video_writer", open_then_interrupted)
    result = _run_pipeline(runner)
    assert writer.released
    assert result["quit_requested"] and result["count"] == 0


def test_a_stop_request_while_the_workers_start_joins_only_the_started_ones(monkeypatch):
    runner = _pipeline_runner(monkeypatch)
    monkeypatch.setattr(runner, "_setup_video_writer", lambda *args: None)
    created = []

    class ThirdStartInterrupted(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def start(self):
            if len([t for t in created if t.ident is not None]) == 2:
                raise KeyboardInterrupt
            super().start()

    monkeypatch.setattr(threading, "Thread", ThirdStartInterrupted)
    result = _run_pipeline(runner)
    started = [t for t in created if t.ident is not None]
    assert len(created) == 5 and len(started) == 2
    assert not any(t.is_alive() for t in started)  # stopped and joined
    assert result["quit_requested"]
