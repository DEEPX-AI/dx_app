"""Machine-readable progress for front ends -- the Python side of progress.hpp.

With DXAPP_PROGRESS=1 the runners call progress_tick(metrics) once per completed frame and
this prints, at most once per second plus once at exit::

    [PROGRESS] frames=N fps=F avg_fps=A read_ms=R pre_ms=P infer_ms=I post_ms=Q

frames   frames completed so far (always first, always present)
fps      frames since the previous line / seconds since it -- the live rate
avg_fps  frames after the first / seconds since the first (start-up and model load excluded)
*_ms     read / preprocess / inference / postprocess average over the same window, from the
         runner's own stage sums (kept in seconds here, printed in milliseconds)

A field that cannot be computed yet is left out; the final line carries run averages.
Without the variable nothing is printed. Same contract as the C++ runners:
tests/progress/test_progress.cpp, tests/python_example/unit/test_progress_py.py.
"""

import atexit
import os
import sys
import threading
import time

_STAGES = (("read_ms", "sum_read"), ("pre_ms", "sum_preprocess"),
           ("infer_ms", "sum_inference"), ("post_ms", "sum_postprocess"))


def _enabled() -> bool:
    v = os.environ.get("DXAPP_PROGRESS", "")
    return bool(v) and v != "0"


class Progress:
    def __init__(self, interval_s: float = 1.0, clock=None, out=None):
        self._interval = interval_s
        self._clock = clock or time.monotonic
        self._out = out
        self._enabled = _enabled()
        self._lock = threading.Lock()
        self._frames = 0
        self._finished = False
        self._printed = False
        self._first_t = 0.0
        self._last_tick_t = 0.0
        self._prev_t = 0.0
        self._prev_frames = 0
        self._raw_sums = None
        self._base_sums = (0.0,) * len(_STAGES)
        self._last_sums = None
        self._prev_sums = None

    @property
    def frames(self) -> int:
        return self._frames

    def tick(self, metrics=None) -> None:
        """One frame finished; ``metrics`` is the runner's dict of running ``sum_*`` (seconds)."""
        if not self._enabled:
            return
        with self._lock:
            self._frames += 1
            n = self._frames
            t = self._clock()
            if n == 1:
                self._first_t = t
            self._last_tick_t = t
            sums = None
            if metrics is not None:
                raw = tuple(float(metrics.get(key, 0.0)) for _, key in _STAGES)
                # A runner that starts new metrics (a new loop) restarts its sums from zero:
                # carry what it had, so the window and the run average keep adding up.
                if self._raw_sums is not None and any(a < b for a, b in zip(raw, self._raw_sums)):
                    self._base_sums = tuple(a + b for a, b in zip(self._base_sums, self._raw_sums))
                self._raw_sums = raw
                sums = tuple(a + b for a, b in zip(self._base_sums, raw))
                self._last_sums = sums
            if self._printed and t - self._prev_t < self._interval:
                return
            parts = [f"[PROGRESS] frames={n}"]
            if self._printed:
                df = n - self._prev_frames
                dt = t - self._prev_t
                if df > 0 and dt > 0:
                    parts.append(f"fps={df / dt:.1f}")
                avg = self._avg_fps(n, t)
                if avg is not None:
                    parts.append(f"avg_fps={avg:.1f}")
                if sums is not None and self._prev_sums is not None and df > 0:
                    parts += self._stage_fields(sums, self._prev_sums, df)
            self._print(" ".join(parts))
            self._printed = True
            self._prev_t = t
            self._prev_frames = n
            self._prev_sums = self._last_sums

    def finish(self) -> None:
        """Final line with the run averages; printed once."""
        if not self._enabled:
            return
        with self._lock:
            if self._finished:
                return
            self._finished = True
            n = self._frames
            parts = [f"[PROGRESS] frames={n}"]
            avg = self._avg_fps(n, self._last_tick_t)
            if avg is not None:
                parts += [f"fps={avg:.1f}", f"avg_fps={avg:.1f}"]
            if self._last_sums is not None and n > 0:
                parts += self._stage_fields(self._last_sums, (0.0,) * len(_STAGES), n)
            self._print(" ".join(parts))

    def _avg_fps(self, n, t):
        if n < 2 or t <= self._first_t:
            return None
        return (n - 1) / (t - self._first_t)

    @staticmethod
    def _stage_fields(now, before, frames):
        fields = []
        for (name, _), a, b in zip(_STAGES, now, before):
            if a - b >= 0:
                fields.append(f"{name}={(a - b) * 1000.0 / frames:.2f}")
        return fields

    def _print(self, line: str) -> None:
        out = self._out or sys.stdout
        out.write(line + "\n")
        out.flush()


_progress = None
_progress_lock = threading.Lock()


def progress() -> Progress:
    """The process-wide counter the runners tick (final line at exit)."""
    global _progress
    with _progress_lock:
        if _progress is None:
            _progress = Progress(1.0)
            atexit.register(_progress.finish)
        return _progress


def progress_tick(metrics=None) -> None:
    progress().tick(metrics)
