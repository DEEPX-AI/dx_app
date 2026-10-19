"""The Python runners print the same [PROGRESS] line as the C++ ones (tests/progress/test_progress.cpp).

"[PROGRESS] frames=N fps=F avg_fps=A read_ms=R pre_ms=P infer_ms=I post_ms=Q" with
DXAPP_PROGRESS=1: at most once per interval plus a final line with run averages. The Python
runners keep their stage sums in seconds; the line is in milliseconds like the C++ one.
"""
import io
import sys
import threading
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "python_example"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utility.progress import Progress  # noqa: E402


class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _fields(line):
    parts = line.split()
    assert parts[0] == "[PROGRESS]", line
    return dict(p.split("=", 1) for p in parts[1:])


def _run(monkeypatch, enabled=True, **kw):
    if enabled:
        monkeypatch.setenv("DXAPP_PROGRESS", "1")
    else:
        monkeypatch.delenv("DXAPP_PROGRESS", raising=False)
    out = io.StringIO()
    return Progress(out=out, **kw), out


def test_nothing_without_the_env_var(monkeypatch):
    p, out = _run(monkeypatch, enabled=False, interval_s=0.0)
    for _ in range(10):
        p.tick()
    p.finish()
    assert out.getvalue() == ""


def test_fps_is_the_live_rate_and_avg_fps_starts_at_the_first_frame(monkeypatch):
    clock = _Clock(10.0)  # model load happened before this
    p, out = _run(monkeypatch, interval_s=1.0, clock=clock)
    p.tick()
    for i in range(1, 11):  # 10 per second
        clock.t = 10.0 + i / 10
        p.tick()
    for i in range(1, 21):  # 20 per second
        clock.t = 11.0 + i / 20
        p.tick()
    p.finish()
    lines = out.getvalue().splitlines()
    assert lines[0] == "[PROGRESS] frames=1"
    assert len(lines) == 4, lines
    assert _fields(lines[1]) == {"frames": "11", "fps": "10.0", "avg_fps": "10.0"}
    assert _fields(lines[2]) == {"frames": "31", "fps": "20.0", "avg_fps": "15.0"}
    assert _fields(lines[3]) == {"frames": "31", "fps": "15.0", "avg_fps": "15.0"}


def test_stage_ms_window_live_and_run_average_at_the_end(monkeypatch):
    clock = _Clock(5.0)
    p, out = _run(monkeypatch, interval_s=0.0, clock=clock)
    m = {"sum_read": 0.0, "sum_preprocess": 0.0, "sum_inference": 0.0, "sum_postprocess": 0.0}
    for i in range(1, 5):
        m["sum_read"] += 0.001
        m["sum_preprocess"] += 0.002
        m["sum_inference"] += 0.010 if i <= 2 else 0.030  # seconds
        m["sum_postprocess"] += 0.003
        clock.t += 0.1
        p.tick(m)
    p.finish()
    lines = out.getvalue().splitlines()
    assert len(lines) == 5
    assert "_ms=" not in lines[0]
    f1 = _fields(lines[1])
    assert (f1["read_ms"], f1["pre_ms"], f1["infer_ms"], f1["post_ms"]) == ("1.00", "2.00", "10.00", "3.00")
    assert _fields(lines[3])["infer_ms"] == "30.00"
    last = _fields(lines[4])
    assert last["frames"] == "4" and last["infer_ms"] == "20.00" and last["read_ms"] == "1.00"


def test_sums_that_restart_for_a_new_loop_keep_adding_up(monkeypatch):
    clock = _Clock(5.0)
    p, out = _run(monkeypatch, interval_s=0.0, clock=clock)
    for v in (0.010, 0.020, 0.010, 0.020):  # the second loop creates new metrics
        clock.t += 0.1
        p.tick({"sum_inference": v})
    p.finish()
    lines = out.getvalue().splitlines()
    assert _fields(lines[2])["infer_ms"] == "10.00"
    assert _fields(lines[4])["infer_ms"] == "10.00"


def test_concurrent_ticks_are_counted_and_lines_stay_whole(monkeypatch):
    p, out = _run(monkeypatch, interval_s=0.0)

    def work():
        for _ in range(500):
            p.tick({"sum_read": 0.0, "sum_preprocess": 0.0, "sum_inference": 0.0, "sum_postprocess": 0.0})

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    p.finish()
    lines = out.getvalue().splitlines()
    assert _fields(lines[-1])["frames"] == "4000"
    assert all(line.startswith("[PROGRESS] frames=") and line.count("[PROGRESS]") == 1 for line in lines)
