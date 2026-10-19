"""Run the graph engine's plain-assert test binary.

The repository has no gtest; C++ behaviour is exercised by pytest driving
built binaries, the same way the CLI tests do.
"""
import os
import re
import sys
from pathlib import Path

import pytest

from conftest import resolve_bin_dir

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.proc import run_bounded  # noqa: E402

BINARY = "graph_engine_test"

# The whole binary, hardware cases included, takes a few minutes; a hung
# NPU wait must fail the test instead of stalling the suite.
TIMEOUT_S = 1800


SUMMARY = re.compile(r"^(\d+) checks, (\d+) failures, (\d+) skipped$")
# OpenCV's own log lines: "[ WARN:0@7.150] global ..." / "[ERROR:0@7.160] ...".
OPENCV_LOG_LINE = re.compile(r"^\[ ?(WARN|ERROR):\d+@", re.M)
# GLib's, which GStreamer logs through: "(graph_engine_test:1234): GStreamer-WARNING **: ...".
GLIB_LOG_LINE = re.compile(r"^\(\S+:\d+\): \S+-(WARNING|CRITICAL) \*\*", re.M)


def run_engine_test(binary, env=None):
    return run_bounded([str(binary)], capture_output=True, text=True,
                       timeout=TIMEOUT_S, env=env)


@pytest.mark.graph
def test_graph_engine_run_is_bounded(tmp_path, monkeypatch):
    """A binary that never exits fails at TIMEOUT_S instead of hanging."""
    import time
    import test_graph_engine as module
    from test_helpers.proc import BoundedTimeout
    stub = tmp_path / BINARY
    stub.write_text("#!/bin/sh\nsleep 60\n")
    stub.chmod(0o755)
    monkeypatch.setattr(module, "TIMEOUT_S", 1)
    started = time.monotonic()
    with pytest.raises(BoundedTimeout):
        module.run_engine_test(stub)
    assert time.monotonic() - started < 30


@pytest.mark.graph
def test_the_log_patterns_see_opencv_and_glib_lines():
    stderr = ("ok\n[ WARN:0@7.150] global cap.cpp (1) x\n"
              "(graph_engine_test:1234): GStreamer-WARNING **: 12:00:00.000: y\n")
    assert OPENCV_LOG_LINE.search(stderr)
    assert GLIB_LOG_LINE.search(stderr)
    assert GLIB_LOG_LINE.search("(a.out:7): GLib-GObject-CRITICAL **: z")
    assert not GLIB_LOG_LINE.search("ok\n[ WARN:0@7.150] global cap.cpp (1) x\n")
    assert not GLIB_LOG_LINE.search("(graph_engine_test:1234): GStreamer-INFO **: z")


@pytest.mark.graph
def test_graph_engine_unit_tests_pass(tmp_path):
    """Failures fail. Skipped hardware cases are reported as a SKIP naming
    them, never hidden inside a PASS (U-35). The run must not leave scratch
    files behind, and OpenCV or GLib must not log on stderr: the tests that
    provoke a failure on purpose silence it for that scope only."""
    binary = resolve_bin_dir() / BINARY
    if not binary.exists():
        pytest.skip("{} not built (configure with -DDXAPP_BUILD_GRAPH_TESTS=ON)".format(BINARY))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    completed = run_engine_test(binary, dict(os.environ, TMPDIR=str(scratch)))
    print(completed.stdout)
    lines = completed.stdout.strip().splitlines()
    summary = SUMMARY.match(lines[-1]) if lines else None
    assert summary, "no summary line:\n" + completed.stdout[-2000:] + completed.stderr[-2000:]
    assert completed.returncode == 0 and summary.group(2) == "0", (
        completed.stdout + completed.stderr)
    assert not OPENCV_LOG_LINE.search(completed.stderr), (
        "OpenCV logged on stderr:\n" + completed.stderr[-3000:])
    assert not GLIB_LOG_LINE.search(completed.stderr), (
        "GLib (GStreamer) logged on stderr:\n" + completed.stderr[-3000:])
    # U-75: a stage whose model has no config.json runs on the factory's
    # defaults without a warning.
    assert "Config file not found" not in completed.stdout + completed.stderr
    # U-35: afterwards TMPDIR is empty, whatever wrote there.
    left = sorted(p.name for p in scratch.iterdir())
    assert left == [], "left behind in TMPDIR: {}".format(left)
    skipped = int(summary.group(3))
    if skipped:
        reasons = [line for line in lines if line.startswith("SKIP ")]
        pytest.skip("{} case(s) skipped:\n{}".format(skipped, "\n".join(reasons)))
