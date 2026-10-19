"""Graceful stop on SIGINT/SIGTERM for the Python example runners.

Mirrors the C++ runners (``src/cpp_example/common/utility/run_dir.hpp``,
``signal_escalation.hpp``):

* The first SIGINT or SIGTERM asks for a graceful stop: the handler raises
  KeyboardInterrupt in the main thread, which every runner turns into "stop
  reading, release the video writer, print the summary".
* Repeats within COALESCE_S (200 ms) of the first request are the same
  request: one Ctrl-C under ``timeout`` reaches the process 2-3 times.
* A SIGINT more than COALESCE_S after the first request - Ctrl-C pressed
  again because the wind-down hangs - ends the process at once with the
  default action (exit status: killed by SIGINT).
* SIGTERM never escalates: ``timeout`` sends it twice and escalates with
  SIGKILL on its own.

Python runs a handler in the main thread between bytecodes, so while the
main thread is inside one long native call, the escalation waits until that
call returns.
"""
import contextlib
import os
import signal
import threading
import time
from typing import Iterator, Optional, Tuple

#: Repeats of an interrupt within this many seconds are the same request.
COALESCE_S = 0.2

_first_request: Optional[float] = None


def classify(signum: int, now: float, first: Optional[float]) -> Tuple[str, Optional[float]]:
    """What a delivery of *signum* at *now* means, given the first request time *first*.

    Returns ``(action, first)``: ``"graceful"`` for the first request,
    ``"terminate"`` for a SIGINT more than COALESCE_S after it, ``"ignore"``
    otherwise; *first* is the (possibly new) first-request time.
    """
    if first is None:
        return "graceful", now
    if signum == signal.SIGINT and now - first > COALESCE_S:
        return "terminate", first
    return "ignore", first


def _handler(signum, _frame) -> None:
    global _first_request
    action, _first_request = classify(signum, time.monotonic(), _first_request)
    if action == "graceful":
        raise KeyboardInterrupt
    if action == "terminate":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)


def install_signal_handlers() -> bool:
    """Install the handler for SIGINT and SIGTERM (main thread only; False elsewhere).

    Keeps an earlier request and the handlers stay installed; a runner uses
    interrupt_scope(), which also starts fresh and restores the previous handlers.
    """
    if threading.current_thread() is not threading.main_thread():
        return False
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return True


@contextlib.contextmanager
def interrupt_scope() -> Iterator[bool]:
    """Install the handlers for one run and restore the previous ones afterwards.

    Each scope starts with no request, so a second run in the same process
    treats its first SIGINT/SIGTERM as a graceful stop again. Yields whether the
    handlers were installed: off the main thread nothing changes (False).
    """
    global _first_request
    if threading.current_thread() is not threading.main_thread():
        yield False
        return
    saved = [(signum, signal.getsignal(signum)) for signum in (signal.SIGINT, signal.SIGTERM)]
    _first_request = None
    install_signal_handlers()
    try:
        yield True
    finally:
        for signum, previous in saved:
            # None: the previous handler was not set from Python and cannot be put back.
            signal.signal(signum, signal.SIG_DFL if previous is None else previous)
        _first_request = None
