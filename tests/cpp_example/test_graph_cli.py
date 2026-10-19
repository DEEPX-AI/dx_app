"""CLI behaviour for the multi-model graph example.

Three tiers, deliberately separated by marker:

  * ``graph``       - no NPU, no .dxnn files. ``--check`` stops before any
                      model is opened and ``--list-models`` reads the
                      generated table, so these run in any checkout.
  * ``graph_e2e``   - real hardware. Runs every shipped sample graph
                      end to end and requires a rendered result image.
  * ``graph_parity``- real hardware. Runs one graph through both executors
                      and requires byte-identical ``--report`` JSON.

The hardware tiers skip for exactly ONE reason: a .dxnn the sample needs is
not downloaded. That is the same rule ``MissingArtifacts()`` applies in
``src/cpp_example/common/graph/test/graph_engine_test.cpp``, and the skip
message names the file and the download command. Anything that goes wrong
AFTER the artifacts are confirmed present - a device that will not open, a
non-zero exit, an empty image - is a failure, never a skip.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT, resolve_bin_dir

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_helpers.proc import run_bounded, wait_until_blocked_opening_a_fifo  # noqa: E402

BINARIES = ["multi_model_graph_sync", "multi_model_graph_async"]
# The recovery command every graph message prints for a .dxnn to fetch. Not
# ./setup.sh --models: setup.sh also re-fetches the sample videos and, under
# its default --force, rm -rf's an assets/videos symlink (get_resource.sh).
DOWNLOAD = "python3 scripts/download_models.py --models"
GRAPHS = "src/cpp_example/multi_model_graph"
GRAPH_DIR = PROJECT_ROOT / "src" / "cpp_example" / "multi_model_graph"
MODEL_DIR = PROJECT_ROOT / "assets" / "models"

# The two-source sample (SP2): cam1 and cam2 feed one tracked od -> reid.
MULTISTREAM = "multistream_od_two_sources.json"
# Every sample with more than one source node: --check lists their streams.
MULTI_SOURCE = (MULTISTREAM, "multistream_people_count.json")

# Every sample graph this example ships. --check uses the full list.
# Hardware tiers use RUNNABLE_SAMPLE_GRAPHS.
SAMPLE_GRAPHS = [
    "fanout_od_seg_pose.json",
    "fanout_od_seg_depth.json",
    "cascade_od_reid_track.json",
    "cascade_od_attr.json",
    "cascade_obb_cls.json",
    "handoff_denoise_od.json",
    "handoff_sr_od_cls.json",
    MULTISTREAM,
    "chain_zerodce_od_pose_emb.json",
    "chain_sr_od_pose.json",
    "hand_cascade.json",
    "dms_headpose.json",
    "logistics_volume.json",
    "zone_intrusion.json",
    "line_counting.json",
    "ppe_zone_check.json",
    "loitering_dwell.json",
    "zone_people_count.json",
    "gallery_known_person.json",
    "driver_attention.json",
    "multistream_people_count.json",
    "logistics_volume_alert.json",
]
# DX-M1 cannot load these together: realesrgan-x2 plus yolov8-n plus a third model.
NPU_WEIGHT_EXCLUDED = (
    "handoff_sr_od_cls.json",
    "chain_sr_od_pose.json",
)
RUNNABLE_SAMPLE_GRAPHS = [g for g in SAMPLE_GRAPHS if g not in NPU_WEIGHT_EXCLUDED]
ONE_SOURCE_SAMPLES = [g for g in SAMPLE_GRAPHS if g not in MULTI_SOURCE]


# Two realesrgan_x2 engines next to yolov8n and resnet50: more than DX-M1's
# device memory holds (the second realesrgan_x2 cannot register its memory
# cache). --check accepts it; only loading can tell.
TWO_SR_GRAPH = {
    "version": 1, "name": "two-sr",
    "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
              {"id": "od", "model": "yolov8n"}, {"id": "cls", "model": "resnet50"},
              {"id": "sr1", "model": "realesrgan_x2"}, {"id": "sr2", "model": "realesrgan_x2"}],
    "edges": [{"from": "cam", "to": "od"},
              {"from": "od", "to": "cls", "roi": {"classes": ["person"]}},
              {"from": "cam", "to": "sr1"}, {"from": "cam", "to": "sr2"}],
}


# One CLI run: a sample graph over an image or a short clip takes seconds;
# a hung NPU wait must fail the test instead of stalling the suite.
RUN_TIMEOUT_S = 600


def run(binary, *args):
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    return run_bounded([str(path)] + list(args), capture_output=True,
                       text=True, cwd=str(PROJECT_ROOT), timeout=RUN_TIMEOUT_S)


def _registry():
    """Registry rows by every name a graph may use: the variant (the model
    key) and the row's own model_name (an old name or an alias, R6)."""
    with open(str(PROJECT_ROOT / "config" / "model_registry.json"),
              encoding="utf-8") as handle:
        entries = json.load(handle)
    rows = {entry["model_name"]: entry for entry in entries}
    rows.update({entry["variant"]: entry for entry in entries if not entry.get("alias_of")})
    return rows


def graph_models(graph):
    """Model names referenced by a sample graph, in declaration order."""
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    return [node["model"] for node in spec["nodes"] if "model" in node]


def missing_artifacts(graph):
    """Mirror of MissingArtifacts() in graph_engine_test.cpp.

    Returns "" when every .dxnn the graph needs is on disk, otherwise a
    reason naming the first absent file and how to get it.
    """
    registry = _registry()
    for name in graph_models(graph):
        entry = registry.get(name)
        if entry is None:
            # NOT a skip: the sample graph names a model that is not even
            # registered, which is a broken sample, not an absent download.
            pytest.fail("{}: model \"{}\" is not in config/model_registry.json"
                        .format(graph, name))
        path = MODEL_DIR / entry["dxnn_file"]
        if not path.is_file():
            return "{} needs {} ({} is absent) - run {} {}".format(
                graph, name, path, DOWNLOAD, name)
    return ""


def _payload_size(payload):
    """How much a stage actually produced. 0 means "nothing decoded".

    For a dense payload this is the NON-ZERO pixel count, not rows*cols.
    rows*cols only says the matrix was allocated, so an all-zero label map
    from a stage that decoded nothing satisfied it - the assertion could
    not fail for the very case it exists to catch.
    """
    if "items" in payload:
        return len(payload["items"])
    if isinstance(payload.get("values"), list):
        return len(payload["values"])
    for key in ("values", "labels", "image"):
        summary = payload.get(key)
        if isinstance(summary, dict):
            return summary["nonzero"]
    return 0


def shifted_sample_frames(count, image="sample_people.jpg"):
    """`image` from sample/img shifted right by 8 px per frame: frame k moves
    8k px. Movement is what makes frame order and tracking observable."""
    import cv2
    import numpy as np
    picture = cv2.imread(str(PROJECT_ROOT / "sample" / "img" / image))
    assert picture is not None, image
    height, width = picture.shape[:2]
    frames = []
    for k in range(count):
        shift = np.float32([[1, 0, 8 * k], [0, 1, 0]])
        frames.append(cv2.warpAffine(picture, shift, (width, height)))
    return frames


def write_moving_video(path, count=8, image="sample_people.jpg"):
    """An MJPG .avi of shifted_sample_frames(count, image) at 10 FPS; returns path."""
    import cv2
    frames = shifted_sample_frames(count, image)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                             (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def write_two_stream_videos(tmp_path):
    """(cam1 video, cam2 video): sample_people.jpg moving for 8 frames and
    sample_person_a1.jpg moving for 5 - two sizes, two lengths."""
    first = write_moving_video(tmp_path / "cam1.avi", count=8)
    second = write_moving_video(tmp_path / "cam2.avi", count=5, image="sample_person_a1.jpg")
    return first, second


def read_in_turn(counts):
    """[(stream, index)] in the order the CLI reads streams of these lengths:
    one frame from each stream still reading, in declaration order."""
    order, index = [], [0] * len(counts)
    while any(index[k] < n for k, (_, n) in enumerate(counts)):
        for k, (name, n) in enumerate(counts):
            if index[k] < n:
                order.append((name, index[k]))
                index[k] += 1
    return order


def one_stream_graph(tmp_path, keep):
    """The two-source sample without every source but `keep`: that stream alone."""
    with open(str(GRAPH_DIR / MULTISTREAM), encoding="utf-8") as handle:
        spec = json.load(handle)
    drop = {n["id"] for n in spec["nodes"] if n.get("type") == "source" and n["id"] != keep}
    spec["nodes"] = [n for n in spec["nodes"] if n["id"] not in drop]
    spec["edges"] = [e for e in spec["edges"] if e["from"] not in drop]
    path = tmp_path / "{}_alone.json".format(keep)
    path.write_text(json.dumps(spec))
    return path


def write_long_video(path, total=1500):
    """An MJPG .avi cycling shifted_sample_frames(40) for `total` frames: far
    more than either executor finishes before a test stops it."""
    import cv2
    frames = shifted_sample_frames(40)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                             (width, height))
    for k in range(total):
        writer.write(frames[k % len(frames)])
    writer.release()
    return path


class LoopingRun(object):
    """A CLI run under Popen, stdout and stderr each pumped on its own thread.
    wait_looping() returns once the CLI has printed (and flushed) its
    "input:" line on stdout, i.e. the frame loop is starting. Content
    assertions use stdout() or stderr(); output() is both, for failure
    messages only (U-46)."""

    def __init__(self, args, env=None):
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True,
                                     cwd=str(PROJECT_ROOT), env=env)
        self.out_lines, self.err_lines = [], []
        self._looping = threading.Event()
        self._readers = [
            threading.Thread(target=self._pump, args=(self.proc.stdout, self.out_lines, True),
                             daemon=True),
            threading.Thread(target=self._pump, args=(self.proc.stderr, self.err_lines, False),
                             daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def _pump(self, stream, lines, watch):
        for line in stream:
            lines.append(line)
            if watch and line.startswith("input:"):
                self._looping.set()

    def stdout(self):
        return "".join(self.out_lines)

    def stderr(self):
        return "".join(self.err_lines)

    def output(self):
        return self.stdout() + self.stderr()

    def wait_looping(self, timeout=120):
        assert self._looping.wait(timeout), "no 'input:' line:\n" + self.output()

    def stop(self, sig, timeout=30):
        assert self.proc.poll() is None, "finished before the signal:\n" + self.output()
        self.proc.send_signal(sig)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail("signal {} did not end the run within {} s".format(sig, timeout))

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for reader in self._readers:
            reader.join(5)


@pytest.mark.graph
def test_looping_run_keeps_stdout_and_stderr_apart(tmp_path):
    """U-46: a content assertion must see the stream the CLI writes to."""
    script = tmp_path / "both.py"
    script.write_text("import sys\nprint('input:  x', flush=True)\n"
                      "print('to stdout', flush=True)\n"
                      "print('to stderr', file=sys.stderr, flush=True)\n")
    run_ = LoopingRun([sys.executable, str(script)])
    try:
        run_.wait_looping(30)
        run_.proc.wait(timeout=30)
    finally:
        run_.close()
    assert run_.stdout() == "input:  x\nto stdout\n"
    assert run_.stderr() == "to stderr\n"
    assert "to stdout" in run_.output() and "to stderr" in run_.output()


# ---------------------------------------------------------------- tier 1


@pytest.mark.graph
def test_run_is_bounded(tmp_path, monkeypatch):
    """A CLI that never exits fails its test at RUN_TIMEOUT_S instead of
    hanging the suite: run() goes through run_bounded."""
    import test_graph_cli as module
    from test_helpers.proc import BoundedTimeout
    stub = tmp_path / BINARIES[0]
    stub.write_text("#!/bin/sh\nsleep 60\n")
    stub.chmod(0o755)
    monkeypatch.setattr(module, "resolve_bin_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "RUN_TIMEOUT_S", 1)
    started = time.monotonic()
    with pytest.raises(BoundedTimeout):
        module.run(BINARIES[0], "--check")
    assert time.monotonic() - started < 30


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_help_lists_every_option(binary):
    result = run(binary, "--help")
    assert result.returncode == 0
    for option in ("--graph", "--input", "--output", "--model-dir",
                   "--max-inflight", "--report", "--check", "--list-models",
                   "--max-jobs-per-stage", "--stall-timeout-ms", "--display",
                   "--report-timing", "--json", "--events", "--loop",
                   "--list-ops"):
        assert option in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_help_describes_max_inflight_as_frames(binary):
    result = run(binary, "--help")
    assert result.returncode == 0
    assert "frames in flight" in result.stdout


@pytest.mark.graph
def test_max_inflight_zero_is_rejected():
    result = run("multi_model_graph_async", "--graph",
                 "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--check", "--max-inflight", "0")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--max-inflight" in result.stderr


ASYNC_ONLY_OPTIONS = ["--max-inflight", "--max-jobs-per-stage", "--stall-timeout-ms"]
NEW_EXECUTOR_OPTIONS = ["--max-jobs-per-stage", "--stall-timeout-ms"]


@pytest.mark.graph
@pytest.mark.parametrize("option", ASYNC_ONLY_OPTIONS)
def test_sync_refuses_async_only_options(option):
    """U-68: the sync executor has no frames in flight; a flag it would ignore is a usage error."""
    result = run("multi_model_graph_sync", "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS), option, "4")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "{} applies to multi_model_graph_async only".format(option) in result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("option", NEW_EXECUTOR_OPTIONS)
@pytest.mark.parametrize("value", ["-1", "1.5", "abc"])
def test_new_executor_options_reject_a_bad_value(option, value):
    result = run("multi_model_graph_async", "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS), option, value)
    assert result.returncode == 2, result.stdout + result.stderr
    assert '{} expects a non-negative integer, got "{}"'.format(option, value) in result.stderr


NUMERIC_OPTIONS = ["--frames", "--max-inflight", "--max-jobs-per-stage",
                   "--stall-timeout-ms"]


@pytest.mark.graph
@pytest.mark.parametrize("option", NUMERIC_OPTIONS)
@pytest.mark.parametrize("value", ["2147483648", "99999999999999999999"])
def test_numeric_options_refuse_a_value_past_the_ceiling(option, value):
    """Final review M3: strtol saturated at LONG_MAX (ERANGE) and was taken
    as given; a timeout that large overflowed on its way to a duration."""
    result = run("multi_model_graph_async", "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS), option, value)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "{} expects an integer from 0 to 2147483647, got \"{}\"".format(
        option, value) in result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("option", NUMERIC_OPTIONS)
def test_numeric_options_accept_the_ceiling(option):
    result = run("multi_model_graph_async", "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS), option, "2147483647")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("option", NEW_EXECUTOR_OPTIONS)
def test_new_executor_options_need_a_value(option):
    result = run("multi_model_graph_async", "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS), option)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "{} needs a value".format(option) in result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("option", NEW_EXECUTOR_OPTIONS)
def test_new_executor_options_accept_zero(option):
    """0 means no limit / wait forever, as in dx_graph."""
    result = run("multi_model_graph_async", "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS), option, "0")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.graph_parity
def test_executor_options_do_not_change_the_report(tmp_path):
    """The options bound waiting and back-pressure; they never change a result."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    video = write_moving_video(tmp_path / "moving.avi")
    plain, tuned = tmp_path / "plain.json", tmp_path / "tuned.json"
    a = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
            "--input", str(video), "--report", str(plain))
    b = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
            "--input", str(video), "--report", str(tuned),
            "--max-jobs-per-stage", "1", "--stall-timeout-ms", "60000")
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    assert plain.read_bytes() == tuned.read_bytes()


@pytest.mark.graph_parity
@pytest.mark.parametrize("graph", ["cascade_od_reid_track.json",
                                   "handoff_denoise_od.json"])
def test_pipelined_async_matches_sync_on_a_moving_video(tmp_path, graph):
    """Several frames in flight must not change one frame's report - and
    reports must come out in frame order. The reid graph adds tracking; the
    hand-off graph (no tracker) runs od on each frame's denoised image,
    handed off on a plain edge."""
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    video = write_moving_video(tmp_path / "moving.avi")

    reports = {}
    for binary, extra in (("multi_model_graph_sync", []),
                          ("multi_model_graph_async", ["--max-inflight", "4"])):
        report = tmp_path / (binary + ".json")
        result = run(binary, "--graph", "{}/{}".format(GRAPHS, graph),
                     "--input", str(video), "--report", str(report), *extra)
        assert result.returncode == 0, result.stdout + result.stderr
        with open(str(report), encoding="utf-8") as handle:
            reports[binary] = json.load(handle)["frames"]
    frames = reports["multi_model_graph_async"]
    assert [f["index"] for f in frames] == list(range(8))
    assert frames == reports["multi_model_graph_sync"]


@pytest.mark.graph_e2e
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
@pytest.mark.parametrize("binary", BINARIES)
def test_first_stop_request_finishes_in_flight_frames_and_writes_the_report(
        tmp_path, binary, sig):
    """One SIGINT or one SIGTERM mid-video is graceful: the input loop stops,
    frames in flight are finished, the report is written and the exit code is
    0. (A later SIGINT terminates instead, and SIGTERM never escalates; both
    are pinned without hardware in common_unit_test.)"""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    total = 1500
    video = write_long_video(tmp_path / "long.avi", total)
    report = tmp_path / "report.json"
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, graph),
                       "--input", str(video), "--report", str(report)])
    try:
        run_.wait_looping()
        time.sleep(1.0)  # some frames into the loop
        run_.stop(sig)
    finally:
        run_.close()
    assert run_.proc.returncode == 0, run_.output()
    with open(str(report), encoding="utf-8") as handle:
        indices = [f["index"] for f in json.load(handle)["frames"]]
    assert 0 < len(indices) < total, len(indices)
    assert indices == list(range(len(indices)))
    assert "{} frames, 0 failed".format(len(indices)) in run_.stdout(), run_.output()


def sigterm_is_caught(pid):
    """Whether /proc says `pid` has a SIGTERM handler installed."""
    try:
        with open("/proc/{}/status".format(pid), encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("SigCgt:"):
                    return (int(line.split()[1], 16) >> (signal.SIGTERM - 1)) & 1 == 1
    except OSError:
        pass
    return False


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_a_stop_during_model_load_is_not_blamed_on_the_input(tmp_path, binary):
    """Final review I2: SIGTERM while the models load stops the run before
    the first frame. That used to print "produced no frames -> check that
    the file is a readable image" and exit 1. The signal goes out the moment
    the CLI's handler is installed, which is just before PrepareGraph."""
    graph = "cascade_od_attr.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    report = tmp_path / "report.json"
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, graph),
                       "--report", str(report)])
    try:
        deadline = time.monotonic() + 60
        while not sigterm_is_caught(run_.proc.pid):
            assert run_.proc.poll() is None, run_.output()
            assert time.monotonic() < deadline, "no SIGTERM handler:\n" + run_.output()
            time.sleep(0.001)
        run_.stop(signal.SIGTERM, timeout=120)
    finally:
        run_.close()
    assert run_.proc.returncode == 0, run_.output()
    assert "interrupted before the first frame" in run_.stderr(), run_.output()
    assert "produced no frames" not in run_.stderr(), run_.output()
    assert "0 frames, 0 failed" in run_.stdout(), run_.output()
    with open(str(report), encoding="utf-8") as handle:
        assert json.load(handle) == {"frames": [], "graph": "person-attributes"}


@pytest.mark.graph_e2e
def test_a_killed_run_leaves_every_finished_frame_in_the_report(tmp_path):
    """U-04: the report is written as frames finish, one write() per frame,
    so even SIGKILL leaves a file that is every finished frame so far and
    becomes valid JSON by appending the closing lines."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    path = resolve_bin_dir() / "multi_model_graph_async"
    if not path.exists():
        pytest.skip("multi_model_graph_async not built")
    video = write_long_video(tmp_path / "long.avi")
    report = tmp_path / "report.json"
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, graph),
                       "--input", str(video), "--report", str(report)])
    try:
        run_.wait_looping()
        time.sleep(2.0)
        growing = report.read_text(encoding="utf-8")
        assert growing.startswith('{\n  "frames": [\n    {'), growing[:120]
        run_.proc.kill()
        run_.proc.wait()
    finally:
        run_.close()
    text = report.read_text(encoding="utf-8")
    assert text.endswith("}"), text[-120:]
    repaired = json.loads(text + '\n  ],\n  "graph": "person-reid"\n}\n')
    indices = [frame["index"] for frame in repaired["frames"]]
    assert indices and indices == list(range(len(indices)))


@pytest.mark.graph_e2e
def test_report_is_finalized_when_an_output_frame_cannot_be_written(tmp_path):
    """A failing --output ends the run with exit 1; the report still closes.
    The output path is an existing directory: its parent exists, so the run
    starts, and the first imwrite fails. (A missing directory is refused
    before the models load - see the test after this one.)"""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report = tmp_path / "report.json"
    target = tmp_path / "out.png"
    target.mkdir()
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--report", str(report), "--output", str(target))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not write {}".format(target) in result.stderr
    with open(str(report), encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["graph"] == "person-reid"
    assert [frame["index"] for frame in payload["frames"]] == [0]


@pytest.mark.graph
@pytest.mark.parametrize("ext", [".png", ".mp4", ".avi"])
def test_output_into_a_missing_directory_fails_before_models_load(tmp_path, ext):
    """Final review I3: not at the first rendered frame behind the models'
    load, and not blamed on the codec."""
    target = tmp_path / "no_such_dir" / ("out" + ext)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_async", "--graph",
                 "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--output", str(target), "--report", str(report))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not write {}: directory {} does not exist".format(
        target, target.parent) in result.stderr, result.stderr
    assert "video writer" not in result.stderr
    assert "graph:" not in result.stdout  # nothing was built
    assert not report.exists()


@pytest.mark.graph_e2e
def test_an_unwritable_report_fails_before_the_first_frame(tmp_path):
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    target = tmp_path / "no_such_dir" / "report.json"
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--report", str(target))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not write {}".format(target) in result.stderr
    assert "input:" not in result.stdout


def run_with_env(binary, env, *args):
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    return run_bounded([str(path)] + list(args), capture_output=True, text=True,
                       cwd=str(PROJECT_ROOT), env=env, timeout=RUN_TIMEOUT_S)


def camera_available():
    import cv2
    if not Path("/dev/video0").exists():
        return False
    capture = cv2.VideoCapture(0)
    ok = capture.isOpened()
    capture.release()
    return ok


# What a camera source says when it does not open (input_factory.hpp).
CAMERA_OPEN_FAILED = "Failed to open camera"


def skip_if_the_camera_did_not_open(output):
    """camera_available() opens and releases the camera, so another process
    can take it before the run opens it: a busy camera, not a failure."""
    if CAMERA_OPEN_FAILED in output:
        pytest.skip("camera /dev/video0 became busy before the run opened it")


def read_video(path):
    """(frame count, fps, (rows, cols) of the last frame) read back with cv2."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    count, shape = 0, None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        shape = frame.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return count, fps, shape


@pytest.mark.graph
@pytest.mark.parametrize("where", ["input_camera", "input_rtsp", "graph_uri"])
def test_live_source_with_image_output_is_refused(tmp_path, where):
    """Review Focus #4: refused before any model loads or any device opens -
    which is why it runs without an NPU, a camera or an RTSP server."""
    graph = "{}/cascade_od_reid_track.json".format(GRAPHS)
    output = tmp_path / "out.png"
    if where == "input_camera":
        args = ["--graph", graph, "--input", "camera:0"]
    elif where == "input_rtsp":
        args = ["--graph", graph, "--input", "rtsp://127.0.0.1:1/none"]
    else:
        with open(str(GRAPH_DIR / "cascade_od_reid_track.json"), encoding="utf-8") as handle:
            spec = json.load(handle)
        for node in spec["nodes"]:
            if node.get("type") == "source":
                node["uri"] = "camera:0"
        live = tmp_path / "live.json"
        live.write_text(json.dumps(spec))
        args = ["--graph", str(live)]
    result = run("multi_model_graph_async", *(args + ["--output", str(output)]))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "is a live source that never ends" in result.stderr
    assert "--frames N" in result.stderr
    # .mkv and .avi first: a killed .mp4 has no index and cannot be read.
    assert "(--output <stem>.mkv, .avi or .mp4)" in result.stderr
    assert not output.exists()
    assert "graph:" not in result.stdout  # nothing was built


@pytest.mark.graph
def test_display_without_a_window_system_is_a_usage_error():
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")}
    result = run_with_env("multi_model_graph_async", env, "--graph",
                          "{}/cascade_od_reid_track.json".format(GRAPHS), "--display")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--display needs a window system" in result.stderr


def unused_x_display():
    """":N" for a display number with no X server socket."""
    for number in range(99, 200):
        if not Path("/tmp/.X11-unix/X{}".format(number)).exists():
            return ":{}".format(number)
    raise AssertionError("no free X display number in :99..:199")


@pytest.mark.graph
@pytest.mark.parametrize("setting", ["display", "qt_xcb"])
def test_display_that_does_not_answer_is_refused_before_models_load(tmp_path, setting):
    """Final review I1: a DISPLAY with no server behind it (or xcb with no
    DISPLAY) made Qt abort after the models loaded, rc 134, report cut off
    mid-array. The probe window opens in a child first: exit 2, nothing
    loaded, no report."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")}
    if setting == "display":
        env["DISPLAY"] = unused_x_display()
        named = "DISPLAY=" + env["DISPLAY"]
    else:
        env["QT_QPA_PLATFORM"] = "xcb"
        named = "QT_QPA_PLATFORM=xcb"
    report = tmp_path / "report.json"
    result = run_with_env("multi_model_graph_async", env, "--graph",
                          "{}/cascade_od_attr.json".format(GRAPHS), "--display",
                          "--report", str(report))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--display could not open a window with {}".format(named) in result.stderr
    assert "graph:" not in result.stdout  # nothing was built
    assert not report.exists()


@pytest.mark.graph_e2e
@pytest.mark.parametrize("ext", [".avi", ".mp4", ".mkv"])
def test_video_output_is_one_file_holding_every_frame(tmp_path, ext):
    """Review Focus #5 (end to end): one file, every reported frame in it, at
    the source's fps and size."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    video = write_moving_video(tmp_path / "moving.avi")  # 8 frames, 10 fps
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / ("result" + ext)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--input", str(video), "--output", str(output), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in out_dir.iterdir()) == ["result" + ext]
    assert "output: {}".format(output) in result.stdout
    count, fps, shape = read_video(output)
    with open(str(report), encoding="utf-8") as handle:
        assert count == len(json.load(handle)["frames"]) == 8
    assert abs(fps - 10) < 0.5, fps
    assert shape == shifted_sample_frames(1)[0].shape[:2]


@pytest.mark.graph_e2e
def test_display_runs_offscreen_and_does_not_change_the_report(tmp_path):
    graph = "handoff_denoise_od.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    video = write_moving_video(tmp_path / "moving.avi")
    plain, shown = tmp_path / "plain.json", tmp_path / "shown.json"
    a = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
            "--input", str(video), "--report", str(plain))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    b = run_with_env("multi_model_graph_async", env, "--graph",
                     "{}/{}".format(GRAPHS, graph), "--input", str(video),
                     "--report", str(shown), "--display")
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    assert "8 frames, 0 failed" in b.stdout
    assert "--display is off" not in b.stderr  # the window really opened
    assert plain.read_bytes() == shown.read_bytes()


@pytest.mark.graph_e2e
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_camera_to_video_stops_on_a_signal_with_a_complete_report(tmp_path, sig):
    """The live case U-04/U-05/U-07 exist for: a camera run with no --frames,
    ended by SIGTERM, leaves a finalized report and a video holding exactly
    the reported frames. One SIGINT or one SIGTERM, both graceful."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    if not camera_available():
        pytest.skip("no camera at /dev/video0")
    path = resolve_bin_dir() / "multi_model_graph_async"
    if not path.exists():
        pytest.skip("multi_model_graph_async not built")
    output, report = tmp_path / "cam.mkv", tmp_path / "report.json"
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, graph),
                       "--input", "camera:0", "--output", str(output),
                       "--report", str(report)])
    try:
        try:
            run_.wait_looping()
        except AssertionError:
            skip_if_the_camera_did_not_open(run_.output())
            raise
        time.sleep(3.0)
        run_.stop(sig)
    finally:
        run_.close()
    assert run_.proc.returncode == 0, run_.output()
    with open(str(report), encoding="utf-8") as handle:
        indices = [frame["index"] for frame in json.load(handle)["frames"]]
    assert indices and indices == list(range(len(indices)))
    count, _, shape = read_video(output)
    assert count == len(indices), (count, len(indices))
    assert len(shape) == 2 and min(shape) > 0, shape  # any camera's frame size


@pytest.mark.graph_e2e
def test_camera_with_frames_writes_that_many_images(tmp_path):
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    if not camera_available():
        pytest.skip("no camera at /dev/video0")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--input", "camera:0", "--frames", "5", "--output", str(out_dir / "cam.png"))
    if result.returncode != 0:
        skip_if_the_camera_did_not_open(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in out_dir.iterdir()) == \
        ["cam_{:06d}.png".format(k) for k in range(5)]


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize("graph", SAMPLE_GRAPHS)
def test_every_sample_graph_validates(binary, graph):
    """--check must pass on every shipped sample without touching the NPU.
    The samples name each model by its variant, so no old-name note is
    printed (Decision 2 stage 1)."""
    result = run(binary, "--check", "{}/{}".format(GRAPHS, graph))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
    assert "is the old name of" not in result.stdout + result.stderr, result.stdout


# Two sources feeding one detector: two streams (SP2). --check only.
TWO_STREAM_CHECK_GRAPH = {
    "version": 1, "name": "two-streams",
    "nodes": [{"id": "cam1", "type": "source", "uri": "sample/img/sample_people.jpg"},
              {"id": "cam2", "type": "source", "uri": "sample/img/sample_person_a1.jpg"},
              {"id": "od", "model": "yolov8n"},
              {"id": "reid", "model": "casvit_t"}],
    "edges": [{"from": "cam1", "to": "od"}, {"from": "cam2", "to": "od"},
              {"from": "od", "to": "reid", "roi": {"classes": ["person"]}}],
}


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_check_lists_one_stream_per_source(tmp_path, binary):
    path = tmp_path / "two.json"
    path.write_text(json.dumps(TWO_STREAM_CHECK_GRAPH))
    result = run(binary, "--check", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("\nstreams: 2 - one per source node, read in turn, one frame from each\n"
            "  cam1      -> od, reid\n"
            "  cam2      -> od, reid\n") in result.stdout, result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("graph", ONE_SOURCE_SAMPLES)
def test_check_of_a_one_source_graph_lists_no_streams(graph):
    """Review Focus #4: one source, --check output unchanged."""
    result = run(BINARIES[0], "--check", "{}/{}".format(GRAPHS, graph))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "streams:" not in result.stdout


@pytest.mark.graph
def test_check_lists_the_two_source_samples_streams():
    result = run(BINARIES[0], "--check", "{}/{}".format(GRAPHS, MULTISTREAM))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "\nstreams: 2 - one per source node, read in turn, one frame from each\n" \
        in result.stdout, result.stdout
    assert "  cam1      -> od, reid\n  cam2      -> od, reid\n" in result.stdout


@pytest.mark.graph
def test_check_lists_the_people_count_samples_streams():
    result = run(BINARIES[0], "--check", "{}/multistream_people_count.json".format(GRAPHS))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("\nstreams: 2 - one per source node, read in turn, one frame from each\n"
            "  cam1      -> od, count, crowded\n"
            "  cam2      -> od, count, crowded\n") in result.stdout, result.stdout


@pytest.mark.graph
def test_check_names_missing_artifacts_with_a_recovery_command(tmp_path):
    result = run(BINARIES[0], "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--model-dir", str(tmp_path))
    assert "missing" in result.stdout.lower()
    # download_models.py, not setup.sh (see DOWNLOAD), into --model-dir.
    assert DOWNLOAD + " " in result.stdout
    assert "--output {}\n".format(tmp_path) in result.stdout, result.stdout
    assert "setup.sh" not in result.stdout, result.stdout
    assert "casvit_t" in result.stdout


@pytest.mark.graph
def test_missing_models_are_named_as_the_model_zoo_spells_them(tmp_path):
    """Pins the CLI text the U-23 refactor must keep (passes before and after)."""
    result = run(BINARIES[0], "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--model-dir", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "  -> {} YoloV8N casvit_t --output {}\n".format(DOWNLOAD, tmp_path) in result.stdout, \
        result.stdout


def swag_download(model_dir):
    return "  -> {} vit_l_p16_512x512_swag --output {}\n".format(DOWNLOAD, model_dir)


@pytest.mark.graph
def test_missing_published_swag_is_offered_a_download(tmp_path):
    """vit-l-p16_512x512_swag is published. A missing file is offered
    download_models.py under the manifest download name, in --check and in a run."""
    graph = _write_graph(tmp_path, [{"id": "od", "model": "yolov8n"},
                                    {"id": "cls", "model": "vit-l-p16_512x512_swag"}])
    empty = tmp_path / "models"
    empty.mkdir()
    check = run(BINARIES[0], "--check", str(graph), "--model-dir", str(empty))
    assert check.returncode == 0, check.stdout + check.stderr
    both = "  -> {} YoloV8N vit_l_p16_512x512_swag --output {}\n".format(DOWNLOAD, empty)
    assert both in check.stdout, check.stdout
    assert "is not published" not in check.stdout
    for binary in BINARIES:
        result = run(binary, "--graph", str(graph), "--model-dir", str(empty))
        assert result.returncode == 1, result.stdout + result.stderr
        assert result.stderr.endswith(both), result.stderr

    alone = _write_graph(tmp_path, [{"id": "cls", "model": "vit-l-p16_512x512_swag"}])
    check = run(BINARIES[0], "--check", str(alone), "--model-dir", str(empty))
    assert check.stdout.endswith(swag_download(empty)), check.stdout
    assert "is not published" not in check.stdout


@pytest.mark.graph
def test_check_is_not_fatal_when_only_artifacts_are_missing(tmp_path):
    """T0 decides the exit code; T1 reports. A user with no models
    downloaded must still be able to check a graph's shape."""
    result = run(BINARIES[0], "--check",
                 "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--model-dir", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.graph
def test_check_rejects_a_bad_graph_with_a_coded_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "version": 1,
        "nodes": [
            {"id": "cam", "type": "source", "uri": "a.jpg"},
            {"id": "seg", "model": "bisenetv2"},
            {"id": "reid", "model": "casvit_t"},
        ],
        "edges": [
            {"from": "cam", "to": "seg"},
            {"from": "seg", "to": "reid", "roi": {}},
        ],
    }))
    result = run(BINARIES[0], "--check", str(bad))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "GRAPH_EDGE" in combined
    # ToString(Shape::kLabelMap) is "labelmap" - the same spelling
    # docs/graph_models.md and the JSON schema use.
    assert "labelmap" in combined
    assert "-> " in combined, "every error must carry a recovery line"


@pytest.mark.graph
@pytest.mark.parametrize("bad_graph,code,needle", [
    ({"version": 9, "nodes": [], "edges": []}, "GRAPH_VERSION", "9"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "no_such_model_at_all"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "MODEL_UNKNOWN", "no_such_model_at_all"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "a", "model": "yolov8n"},
                {"id": "b", "model": "yolov8n"}],
      "edges": [{"from": "cam", "to": "a"},
                {"from": "a", "to": "b"},
                {"from": "b", "to": "a"}]},
     "GRAPH_CYCLE", "cycle"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "lost", "model": "resnet50"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "GRAPH_ORPHAN", "lost"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n", "prompt": "a person"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "GRAPH_RESERVED", "prompt"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "cls", "model": "resnet50"}],
      "edges": [{"from": "cam", "to": "od"},
                {"from": "od", "to": "cls", "roi": {}},
                {"from": "od", "to": "cls", "roi": {}}]},
     "GRAPH_SCHEMA", "duplicate edge"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "cls", "model": "resnet50"}],
      "edges": [{"from": "cam", "to": "od"},
                {"from": "cam", "to": "cls"},
                {"from": "od", "to": "cls", "roi": {}}]},
     "GRAPH_EDGE", "both a full frame and ROI"),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "face", "model": "casvit_t"}],
      "edges": [{"from": "cam", "to": "od"},
                {"from": "od", "to": "face", "roi": {"align": "face5"}}]},
     "GRAPH_ALIGN", "landmarks"),
    ({"version": 1.5, "nodes": [], "edges": []}, "GRAPH_SCHEMA", "whole number"),
    ({"version": 1, "name": 5,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "GRAPH_SCHEMA", '"name" must be a string'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "cls", "model": "resnet50"}],
      "edges": [{"from": "cam", "to": "od"},
                {"from": "od", "to": "cls", "roi": {"max": 0}}]},
     "GRAPH_SCHEMA", '"max" must be a whole number of at least 1'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n", "track": {"algo": "sort"}}],
      "edges": [{"from": "cam", "to": "od"}]},
     "GRAPH_SCHEMA", 'accepted "algo" values: iou'),
])
def test_each_error_class_names_where_what_and_how(tmp_path, bad_graph, code,
                                                   needle):
    """The error message is the product: code, location, fact, recovery."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad_graph))
    result = run(BINARIES[0], "--check", str(path))
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined
    assert "ERROR [{}]".format(code) in combined, combined
    assert needle in combined, combined


@pytest.mark.graph
def test_every_bad_value_in_one_roi_is_reported_together(tmp_path):
    """Like unknown keys: one round trip names every bad value of one object."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "yolov8n"},
                  {"id": "cls", "model": "resnet50"}],
        "edges": [{"from": "cam", "to": "od"},
                  {"from": "od", "to": "cls",
                   "roi": {"pad": -1, "max": 2.7, "min_score": 35}}]}))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "3 invalid values:" in result.stderr, result.stderr
    for key in ('"pad"', '"max"', '"min_score"'):
        assert key in result.stderr, result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("graph,key,suggestion", [
    # The reviewer's own probe, one typo at a time. The two-typo form of
    # it now reports both keys in one message and is pinned separately by
    # test_every_unknown_key_is_reported_in_one_message.
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "reid", "model": "casvit_t"}],
      "edges": [{"from": "cam", "to": "od"},
                {"from": "od", "to": "reid",
                 "roi": {"classez": ["person"]}}]},
     "classez", 'did you mean "classes"'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"},
                {"id": "reid", "model": "casvit_t"}],
      "edges": [{"from": "cam", "to": "od"},
                {"from": "od", "to": "reid", "roi": {"paddng": 0.5}}]},
     "paddng", 'did you mean "pad"'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "modle": "yolov8n"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "modle", 'did you mean "model"'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n", "track": {"maxage": 30}}],
      "edges": [{"from": "cam", "to": "od"}]},
     "maxage", 'did you mean "max_age"'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"}],
      "edges": [{"from": "cam", "to": "od", "rio": {}}]},
     "rio", 'did you mean "roi"'),
    ({"version": 1, "nodez": [], "edges": []},
     "nodez", 'did you mean "nodes"'),
])
def test_check_rejects_an_unknown_key_and_names_it(tmp_path, graph, key,
                                                   suggestion):
    """A silently dropped key is worse than an error: the graph runs and
    quietly does something other than what was asked."""
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(graph))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "ERROR [GRAPH_SCHEMA]" in combined, combined
    assert 'unknown key "{}"'.format(key) in combined, combined
    assert suggestion in combined, combined
    assert "accepted keys:" in combined, combined


@pytest.mark.graph
@pytest.mark.parametrize("graph,needles", [
    # A mistyped REQUIRED key. The old behaviour already errored here, so
    # these pin the suggestion and the label, not the fact of an error.
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"}],
      "edges": [{"form": "cam", "to": "od"}]},
     ['unknown key "form"', 'did you mean "from"', '?->"od"']),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"di": "od", "model": "yolov8n"}],
      "edges": [{"from": "cam", "to": "od"}]},
     ['unknown key "di"', 'did you mean "id"', '(model "yolov8n")']),
])
def test_a_mistyped_required_key_is_answered_as_a_typo(tmp_path, graph,
                                                       needles):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(graph))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    for needle in needles:
        assert needle in combined, combined
    assert "missing required key" not in combined, combined


@pytest.mark.graph
@pytest.mark.parametrize("graph,key,label", [
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"model": "yolov8n"}],
      "edges": []},
     "id", '(model "yolov8n")'),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"}],
      "edges": [{"from": "cam"}]},
     "to", '"cam"->?'),
])
def test_a_genuinely_missing_key_keeps_its_label_and_gains_a_remedy(
        tmp_path, graph, key, label):
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(graph))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert 'missing required key "{}"'.format(key) in combined, combined
    assert label in combined, combined
    assert '-> add "{}"'.format(key) in combined, combined


@pytest.mark.graph
def test_every_unknown_key_is_reported_in_one_message(tmp_path):
    """Sorted iteration order hid the second typo behind the first."""
    path = tmp_path / "typos.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "yolov8n"},
                  {"id": "reid", "model": "casvit_t"}],
        "edges": [{"from": "cam", "to": "od"},
                  {"from": "od", "to": "reid",
                   "roi": {"paddng": 0.5, "classez": ["person"]}}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "2 unknown keys" in combined, combined
    assert '"classes" for "classez"' in combined, combined
    assert '"pad" for "paddng"' in combined, combined


@pytest.mark.graph
@pytest.mark.parametrize("type_value,needle", [
    # The dangerous case: a node that also carries "model" used to be
    # silently accepted, and "model" is what a hand-editor writes.
    ("model", 'unknown node type "model"'),
    ("sink", 'unknown node type "sink"'),
    ("sourse", 'did you mean "source"'),
])
def test_check_rejects_an_unknown_node_type(tmp_path, type_value, needle):
    path = tmp_path / "type.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "type": type_value, "model": "yolov8n"}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0, result.stdout + result.stderr
    assert needle in (result.stdout + result.stderr)


@pytest.mark.graph
def test_schema_key_is_accepted_but_comment_keys_are_not(tmp_path):
    """$schema is inert - nothing in the schema is within edit distance 2
    of it - so tolerating it cannot mask a typo. The tolerance stops
    there."""
    good = tmp_path / "schema.json"
    good.write_text(json.dumps({
        "$schema": "https://example/graph.schema.json",
        "version": 1, "name": "s",
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "yolov8n"}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(good))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout

    bad = tmp_path / "comment.json"
    bad.write_text(json.dumps({
        "version": 1, "_comment": "why this graph exists",
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "yolov8n"}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(bad))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "_comment" in (result.stdout + result.stderr)


@pytest.mark.graph
def test_check_offers_no_suggestion_it_cannot_justify(tmp_path):
    """A wrong guess is worse than none; the accepted list still prints."""
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "yolov8n", "zzzzzzzz": 1}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "did you mean" not in combined, combined
    assert "accepted keys:" in combined, combined


@pytest.mark.graph
def test_params_keys_stay_open(tmp_path):
    """params carries model parameter names, not schema keys."""
    path = tmp_path / "params.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "yolov8n",
                   "params": {"score_threshold": 0.4, "anything_at_all": 2}}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.graph
def test_a_string_for_a_numeric_param_fails_check(tmp_path):
    """U-62: a string reaches the parser now, and ValidateGraph refuses it
    for a key the model reads as a number. A GraphError exits 1."""
    graph = tmp_path / "bad.json"
    graph.write_text('{"version":1,"nodes":[{"id":"cam","type":"source","uri":"a.jpg"},'
                     '{"id":"od","model":"yolov8n","params":{"score_threshold":"0.5"}}],'
                     '"edges":[{"from":"cam","to":"od"}]}', encoding="utf-8")
    result = run("multi_model_graph_sync", "--check", str(graph))
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'node "od" "params": "score_threshold" must be a number, got "0.5"' in result.stderr
    # "yolov8n" is an old name (R6): the hint names the variant that runs.
    assert 'model "yolov8-n_640x640" reads text only for: class_names' in result.stderr


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_class_names_in_params_rename_classes_and_feed_roi(tmp_path, binary):
    if missing_artifacts("cascade_od_attr.json"):
        pytest.skip(missing_artifacts("cascade_od_attr.json"))
    graph = tmp_path / "rename.json"
    graph.write_text(json.dumps({
        "version": 1, "name": "rename",
        "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
                  {"id": "od", "model": "yolov8n", "params": {"class_names": ["pedestrian"]}},
                  {"id": "attr", "model": "deepmar_resnet50"}],
        "edges": [{"from": "cam", "to": "od"},
                  {"from": "od", "to": "attr", "roi": {"classes": ["pedestrian"], "max": 3}}]}),
        encoding="utf-8")
    report = tmp_path / "r.json"
    result = run(binary, "--graph", str(graph), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.load(open(str(report), encoding="utf-8"))["frames"][0]
    names = {item["class_name"] for item in frame["nodes"]["od"]["payload"]["items"]}
    assert "pedestrian" in names and "person" not in names
    assert len(frame["roi_nodes"]["attr"]) >= 1


@pytest.mark.graph
def test_unreadable_graph_file_is_a_coded_error(tmp_path):
    result = run(BINARIES[0], "--check", str(tmp_path / "does_not_exist.json"))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "ERROR [GRAPH_SCHEMA]" in combined
    assert "does_not_exist.json" in combined


@pytest.mark.graph
def test_unknown_option_is_rejected_with_usage():
    result = run(BINARIES[0], "--no-such-option")
    assert result.returncode == 2
    assert "--graph" in (result.stdout + result.stderr)


@pytest.mark.graph
def test_no_graph_is_rejected_rather_than_silently_doing_nothing():
    result = run(BINARIES[0])
    assert result.returncode == 2
    assert "--graph" in (result.stdout + result.stderr)


@pytest.mark.graph
def test_list_models_filters_by_capability():
    result = run(BINARIES[0], "--list-models", "--consumes", "roi")
    assert result.returncode == 0
    assert "casvit-t_224x224" in result.stdout
    assert "bisenetv2_1024x2048" not in result.stdout


@pytest.mark.graph
def test_list_models_filters_by_shape():
    result = run(BINARIES[0], "--list-models", "--produces", "keypoints")
    assert result.returncode == 0
    assert "yolov8-s-pose_640x640" in result.stdout
    assert "yolov8-n_640x640 " not in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("option,value,expected", [
    ("--consumes", "nonsense", "roi, frame or either"),
    ("--produces", "bogus_shape", "--produces expects one of"),
    # A real Shape name spelled the way another part of the tree spells it
    # is still not a Shape name here, and must say so rather than printing
    # an empty table.
    ("--produces", "label map", "--produces expects one of"),
])
def test_list_models_rejects_an_unknown_filter_value(option, value, expected):
    """Both filters behave the same way. An empty table and exit 0 reads as
    "no model matches", which is the wrong answer to a typo."""
    result = run(BINARIES[0], "--list-models", option, value)
    assert result.returncode == 2, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert expected in combined, combined


@pytest.mark.graph
@pytest.mark.parametrize("shape", [
    "frame", "boxes", "obboxes", "instances", "keypoints", "labelmap",
    "densemap", "image", "scores", "vector", "boxes3d",
])
def test_list_models_accepts_every_shape_name(shape):
    """The names --produces accepts are exactly the ones ToString prints,
    so the error message above can list them and be right."""
    result = run(BINARIES[0], "--list-models", "--produces", shape)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.graph
def test_list_models_names_the_ports():
    result = run("multi_model_graph_sync", "--list-models")
    assert result.returncode == 0, result.stderr
    rows = {line.split()[0]: line for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("-") and not line.startswith("model ")}
    assert rows["yolopv2_384x640"].endswith("  ports: drivable=labelmap lane=labelmap")
    assert rows["superpoint_480x640"].endswith("  ports: descriptors=densemap")
    assert rows["mediapipe-hands-lite_224x224"].endswith("  ports: handedness=scores")
    assert rows["3ddfa-v2_mobilenetv1_120x120"].endswith("  ports: pose=vector")
    assert rows["ppmatting-hrnet-w48-composition_512x512"].endswith("  ports: alpha=densemap")
    assert rows["eigenplaces-resnet18_512x512"].endswith("  ports: matches=scores")
    assert ("ports:" not in rows["yolov8-s-pose_640x640"]
            and "ports:" not in rows["yolov8-n_640x640"])


@pytest.mark.graph
def test_generated_docs_match_the_binary():
    """docs/graph_models.md is generated (and guarded by
    scripts/check_graph_models_doc.py), so it cannot disagree with
    --list-models."""
    result = run(BINARIES[0], "--list-models")
    assert result.returncode == 0
    with open(str(PROJECT_ROOT / "docs" / "graph_models.md"),
              encoding="utf-8") as handle:
        docs = handle.read()
    listed = 0
    for line in result.stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if name and not name.startswith("-") and name != "model":
            assert "`{}`".format(name) in docs, name
            listed += 1
    assert listed > 300, "--list-models printed only {} models".format(listed)


@pytest.mark.graph
@pytest.mark.parametrize("graph", SAMPLE_GRAPHS)
def test_every_sample_graph_is_retargetable_without_a_rebuild(tmp_path, graph):
    """The headline claim, in test form: the model name is data.

    Rewriting a sample's detector to a different model and re-running
    --check against the SAME binary must validate. If this ever needs a
    rebuild, the project has not met its goal.
    """
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    # (old name, variant) -> the variant swapped in: the samples use old
    # names until they are renamed, and either spelling is swapped.
    pairs = [(("yolov8n", "yolov8-n_640x640"), "yolo26-s_640x640"),
             (("yolo26l_obb", "yolo26-l-obb_1024x1024"), "yolo26-n-obb_1024x1024"),
             (("bisenetv2", "bisenetv2_1024x2048"), "pidnet-s_1024x2048"),
             (("resnet50", "resnet50_224x224"), "mobilenetv2_224x224"),
             (("casvit_t", "casvit-t_224x224"), "casvit-m_224x224"),
             (("deepmar_resnet50", "deepmar_resnet50_224x224"), "deepmar_resnet18_224x224"),
             (("fastdepth_1", "fastdepth_224x224"), "scdepthv3_256x320"),
             (("yolov8s_pose", "yolov8-s-pose_640x640"), "yolo26-n-pose_640x640"),
             (("yolov5n", "yolov5-n_640x640"), "yolov5-s_640x640"),
             (("dncnn_color_blind", "dncnn-color_512x512"), "dncnn-15_512x512"),
             (("realesrgan_x2", "realesrgan-x2_192x192"), "realesrgan-x4_192x192"),
             (("mediapipe_hand_detector_1", "mediapipe-hand-detector_192x192"),
              "scrfd-500m_640x640"),
             (("scrfd500m", "scrfd-500m_640x640"), "yolov8-n_640x640"),
             (("yolo26n", "yolo26-n_640x640"), "yolov8-n_640x640")]
    swaps = {name: target for names, target in pairs for name in names}
    swapped = False
    for node in spec["nodes"]:
        if node.get("model") in swaps:
            node["model"] = swaps[node["model"]]
            swapped = True
    assert swapped, "no swappable model in " + graph
    path = tmp_path / graph
    path.write_text(json.dumps(spec))
    # The question is validity, not what is downloaded: an empty model dir
    # lists every .dxnn as missing (not an error), where a broken file in
    # assets/models would make --check exit 1 (MODEL_LOAD).
    no_models = tmp_path / "no_models"
    no_models.mkdir()
    result = run(BINARIES[0], "--check", str(path), "--model-dir", str(no_models))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.graph
def test_check_rejects_a_plain_edge_that_would_discard_its_producer(tmp_path):
    """A detector feeding a PLAIN edge: the boxes are discarded and the
    consumer would silently run on the raw source frame, so it stays
    refused - unlike an image producer (Task 2: image hand-off), a boxes
    producer's output cannot be handed off wholesale, only cropped."""
    path = tmp_path / "discard.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "first", "model": "yolov8n"},
                  {"id": "second", "model": "bisenetv2"}],
        "edges": [{"from": "cam", "to": "first"},
                  {"from": "first", "to": "second"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "ERROR [GRAPH_EDGE]" in combined, combined
    assert "would be discarded" in combined, combined
    assert "add \"roi\": {} to crop" in combined, combined


@pytest.mark.graph
def test_check_accepts_an_image_hand_off_on_a_plain_edge(tmp_path):
    """Task 2 (image hand-off on a plain edge): this was the case that made
    the old rejection urgent - "denoise, then detect" is the most obvious
    thing a JSON-only user composes, and 13 registry models produce
    "image". It is now accepted: the denoiser's own output image is handed
    off across the plain edge to the next model."""
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "first", "model": "dncnn_15"},
                  {"id": "second", "model": "bisenetv2"}],
        "edges": [{"from": "cam", "to": "first"},
                  {"from": "first", "to": "second"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.graph
def test_plain_edge_from_non_image_is_rejected(tmp_path):
    """Only an image can be handed off on a plain edge. A label map is
    neither an image nor something to crop from, so it gets the hand-off
    text, not the boxes producer's "would be discarded" one."""
    path = tmp_path / "non_image.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "seg", "model": "bisenetv2"},
                  {"id": "od", "model": "yolov8n"}],
        "edges": [{"from": "cam", "to": "seg"},
                  {"from": "seg", "to": "od"}],
    }))
    result = run("multi_model_graph_sync", "--check", str(path))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "ERROR [GRAPH_EDGE]" in result.stderr, result.stderr
    assert "\"seg\" produces labelmap" in result.stderr, result.stderr
    assert "only an image can be handed off on a plain edge" in result.stderr, \
        result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("graph", SAMPLE_GRAPHS)
def test_the_rule_against_discarding_does_not_reject_a_valid_graph(graph):
    """A rule that rejects a legitimate composition is worse than the
    silence it replaces. The shipped samples cover both legitimate plain
    edges: from a source, and from an image producer (the hand-off
    samples)."""
    result = run(BINARIES[0], "--check", "{}/{}".format(GRAPHS, graph))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("typo,suggestion", [
    ("Resnet50", 'did you mean "resnet50_224x224"'),
    # The model key is the variant (R6), and users arrive with an old name
    # or the model zoo's spelling: the nearest of those names the variant.
    ("YoloV8N", 'did you mean "yolov8-n_640x640"'),
    ("FastSAM-s", 'did you mean "fastsam-s_1024x1024"'),
    ("yolo26l-obb", 'did you mean "yolo26-l-obb_1024x1024"'),
])
def test_an_unknown_model_name_gets_a_suggestion(tmp_path, typo, suggestion):
    """The model name is the one string the README tells users to edit."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": typo}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "ERROR [MODEL_UNKNOWN]" in combined, combined
    assert suggestion in combined, combined


@pytest.mark.graph
def test_a_far_off_model_name_gets_no_guess(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                  {"id": "od", "model": "zzzzzzzzzzzz"}],
        "edges": [{"from": "cam", "to": "od"}],
    }))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "did you mean" not in combined, combined
    assert "--list-models" in combined, combined


README = PROJECT_ROOT / "src" / "cpp_example" / "multi_model_graph" / "README.md"
DX_GRAPH_README = PROJECT_ROOT / "src" / "bindings" / "python" / "dx_graph" / "README.md"


@pytest.mark.graph
def test_readme_options_table_lists_every_help_option():
    result = run(BINARIES[1], "--help")
    assert result.returncode == 0
    options = sorted(set(re.findall(r"^\s+(--[a-z][a-z-]*)", result.stdout, re.M)))
    table = README.read_text(encoding="utf-8").split("## Options", 1)[1].split("\n---", 1)[0]
    missing = [option for option in options if "`{}".format(option) not in table]
    assert not missing, missing


@pytest.mark.graph
def test_readme_error_table_lists_every_code():
    shape_cpp = (PROJECT_ROOT / "src" / "cpp_example" / "common" / "graph"
                 / "shape.cpp").read_text(encoding="utf-8")
    body = shape_cpp.split("ToString(GraphErrorCode code)", 1)[1].split('return "UNKNOWN"', 1)[0]
    codes = re.findall(r'return "([A-Z_]+)";', body)
    assert "MODEL_LOAD" in codes
    text = README.read_text(encoding="utf-8")
    for code in codes:
        assert "| `{}` |".format(code) in text, code
    words = {11: "eleven", 12: "twelve", 13: "thirteen"}
    assert "There are {} codes.".format(words[len(codes)]) in text


@pytest.mark.graph
def test_docs_say_every_cpp_binary_handles_sigterm():
    assert "does not catch SIGTERM" not in README.read_text(encoding="utf-8")
    for name in ("03_DX-APP_CPP_Example_Usage_Guide.md", "09_DX-APP_Project_Overview.md"):
        text = (PROJECT_ROOT / "docs" / "source" / "docs" / name).read_text(encoding="utf-8")
        paragraph = text.split("**Signal Handling**", 1)[1].split("\n**", 1)[0]
        assert "multi_model_graph" in paragraph, name
        assert "SIGTERM is always graceful" in paragraph, name


@pytest.mark.graph
def test_readmes_document_model_load_non_finite_values_and_device_memory():
    for path in (README, DX_GRAPH_README):
        text = path.read_text(encoding="utf-8")
        assert "MODEL_LOAD" in text, path
        assert '"NaN"' in text and '"Infinity"' in text and '"-Infinity"' in text, path
        assert "Device memory" in text, path
        assert "realesrgan-x2_192x192" in text, path
        assert "cannot hold a second realesrgan_x2" not in text, path  # M2: the old names


@pytest.mark.graph
def test_readmes_document_several_sources():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Several sources: one stream each", 1)[1].split("\n---", 1)[0]
    for needle in ("--input <source_id>=<uri>", "<stem>_<source_id><ext>",
                   "<stem>_<source_id>_<index><ext>", '"stream"', "in turn",
                   "--frames", "--max-inflight", "--check",
                   "one window, `multi_model_graph`, holding every stream",
                   "labelled with its source id",
                   "one tracker per stream"):
        assert needle in section, needle
    assert "multistream_od_two_sources.json" in text
    # The parity claim counts what SAMPLE_GRAPHS compares against what ships.
    words = {11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
             16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
             21: "twenty-one", 22: "twenty-two"}
    shipped = len(list(GRAPH_DIR.glob("*.json")))
    if len(SAMPLE_GRAPHS) == shipped:
        claim = "all {} shipped sample graphs".format(words[shipped])
    else:
        claim = "{} of the {} shipped sample graphs".format(
            words[len(SAMPLE_GRAPHS)], words[shipped])
    assert claim in " ".join(text.split()), claim
    assert "more than one full-frame image into one node in one stream" in text
    py = DX_GRAPH_README.read_text(encoding="utf-8")
    for needle in ("sources=", "run(frame, stream=None)", "| `stream` |", "g.streams",
                   "(source_id, frame)"):
        assert needle in py, needle


@pytest.mark.graph
def test_readme_lists_every_shipped_sample_graph():
    """The samples table names exactly the graph files that ship next to it."""
    text = README.read_text(encoding="utf-8")
    table = text.split("Shipped samples, next to the CLI sources", 1)[1].split("\n\n", 2)[1]
    listed = set(re.findall(r"^\| `([^`]+\.json)` \|", table, re.M))
    assert listed == {p.name for p in GRAPH_DIR.glob("*.json")}


@pytest.mark.graph
def test_driver_attention_decides_on_head_pose_numbers():
    """A condition reads a cpu node's records: the head pose's yaw and pitch."""
    code, out = _check_json(GRAPH_DIR / "driver_attention.json")
    assert code == 0 and out["ok"] is True and out["errors"] == [], out
    with open(str(GRAPH_DIR / "driver_attention.json"), encoding="utf-8") as handle:
        spec = json.load(handle)
    conditions = {n["id"]: n["params"] for n in spec["nodes"] if n.get("op") == "condition"}
    assert sorted(conditions) == ["head_down", "yaw_high", "yaw_low"]
    assert {(p["value"], p["event"]) for p in conditions.values()} == {("yaw", "looking_away"),
                                                                        ("pitch", "head_down")}
    fed = {e["to"] for e in spec["edges"] if e["from"] == "headpose" and e.get("carry") == "result"}
    assert fed == set(conditions)


@pytest.mark.graph
def test_logistics_volume_alert_decides_on_the_summed_volume():
    code, out = _check_json(GRAPH_DIR / "logistics_volume_alert.json")
    assert code == 0 and out["ok"] is True and out["errors"] == [], out
    with open(str(GRAPH_DIR / "logistics_volume_alert.json"), encoding="utf-8") as handle:
        spec = json.load(handle)
    (load,) = [n for n in spec["nodes"] if n.get("op") == "condition"]
    assert load["params"]["value"] == "volume_proxy" and load["params"]["reduce"] == "sum"
    assert load["params"]["event"] == "load_high"
    assert {"from": "volume", "to": load["id"], "carry": "result"} in spec["edges"]


@pytest.mark.graph
def test_docs_say_a_stop_request_finishes_every_stream():
    for name in ("03_DX-APP_CPP_Example_Usage_Guide.md", "09_DX-APP_Project_Overview.md"):
        text = (PROJECT_ROOT / "docs" / "source" / "docs" / name).read_text(encoding="utf-8")
        paragraph = text.split("**Signal Handling**", 1)[1].split("\n**", 1)[0]
        assert "in every stream" in paragraph, name


# ---------------------------------------------------------------- tier 2


@pytest.mark.graph_e2e
@pytest.mark.parametrize("graph", RUNNABLE_SAMPLE_GRAPHS)
def test_graph_runs_and_saves_a_result_image(tmp_path, graph):
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    output = tmp_path / (graph.replace(".json", ".png"))
    report = tmp_path / (graph.replace(".json", "_report.json"))
    result = run("multi_model_graph_async", "--graph",
                 "{}/{}".format(GRAPHS, graph), "--output", str(output),
                 "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    sources = [node["id"] for node in spec["nodes"] if node.get("type") == "source"]
    # A rendered image proves nothing on its own - an untouched copy of the
    # source would pass. The report proves every model node produced a
    # result.
    with open(str(report), encoding="utf-8") as handle:
        payload = json.load(handle)
    if len(sources) == 1:
        assert output.exists() and output.stat().st_size > 0
        frames = payload["frames"][:1]
    else:
        # One image per stream frame: <stem>_<source>_<index6><ext> (spec R6).
        assert not output.exists()
        for source in sources:
            image = output.with_name("{}_{}_{:06d}{}".format(
                output.stem, source, 0, output.suffix))
            assert image.exists() and image.stat().st_size > 0, image
        frames = payload["frames"]
        assert [f["stream"] for f in frames] == sources
    expected = {node["id"] for node in spec["nodes"] if "model" in node}
    for frame in frames:
        produced = set(frame["nodes"].keys()) | set(frame["roi_nodes"].keys())
        assert expected <= produced, "no result for {}".format(expected - produced)
        assert not frame["error"], frame["error"]

        # An empty result set would satisfy everything above, so require that
        # each model node actually decoded something. A ROI node's list is
        # keyed even when its producer found nothing, which is exactly the
        # hole this closes.
        for node in spec["nodes"]:
            if "model" not in node:
                continue
            node_id = node["id"]
            if node_id in frame["nodes"]:
                assert _payload_size(frame["nodes"][node_id]["payload"]) > 0, \
                    "{}: node {} produced an empty payload".format(graph, node_id)
            else:
                crops = frame["roi_nodes"][node_id]
                assert crops, "{}: node {} received no crops".format(graph, node_id)
                for crop in crops:
                    assert _payload_size(crop["payload"]) > 0, \
                        "{}: node {} produced an empty crop result".format(
                            graph, node_id)


@pytest.mark.graph_e2e
def test_model_dir_without_the_artifacts_fails_with_a_recovery_command(tmp_path):
    """The fatal half of the T1 check: a real run, not --check."""
    result = run("multi_model_graph_async", "--graph",
                 "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--model-dir", str(tmp_path), "--output",
                 str(tmp_path / "out.png"))
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "MODEL_MISSING" in combined or "missing" in combined.lower()
    assert DOWNLOAD + " " in combined
    assert "setup.sh" not in combined
    assert not (tmp_path / "out.png").exists()


@pytest.mark.graph_e2e
def test_a_model_the_npu_cannot_hold_is_model_load(tmp_path):
    absent = missing_artifacts("handoff_sr_od_cls.json")  # the same four models
    if absent:
        pytest.skip(absent)
    path = tmp_path / "two_sr.json"
    path.write_text(json.dumps(TWO_SR_GRAPH))
    result = run("multi_model_graph_async", "--graph", str(path))
    if result.returncode == 0:
        pytest.skip("this NPU holds a second realesrgan_x2 next to yolov8n and resnet50")
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'ERROR [MODEL_LOAD] node "sr' in result.stderr, result.stderr
    assert 'model "realesrgan_x2"' in result.stderr
    # Once: MODEL_LOAD names it, the runtime's text follows unprefixed.
    assert result.stderr.count('model "realesrgan_x2"') == 1, result.stderr
    assert 'realesrgan_x2" failed to open' not in result.stderr
    assert "device memory may be full" in result.stderr
    assert "ERROR: model" not in result.stderr  # no longer the untyped fallback


@pytest.mark.graph_e2e
def test_driver_attention_conditions_read_the_head_pose_of_their_frame(tmp_path):
    graph = "driver_attention.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_sync", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.loads(report.read_text(encoding="utf-8"))["frames"][0]
    poses = frame["nodes"]["headpose"]["payload"]["items"]
    assert len(poses) == 1, "sample_driver_monitoring.png shows one face: {}".format(poses)
    pose = poses[0]["numbers"]
    for node, name in (("yaw_high", "yaw"), ("yaw_low", "yaw"), ("head_down", "pitch")):
        numbers = frame["nodes"][node]["payload"]["items"][0]["numbers"]
        assert numbers["value"] == pytest.approx(pose[name]), node
    # The sample driver looks ahead (yaw about 14, pitch about -6 degrees): no condition holds.
    assert -30 < pose["yaw"] < 30 and pose["pitch"] > -20, pose
    assert not any(frame["nodes"][n]["payload"]["items"][0]["numbers"]["active"]
                   for n in ("yaw_high", "yaw_low", "head_down"))
    assert "events" not in frame


@pytest.mark.graph_e2e
def test_dms_headpose_gives_every_face_its_pose(tmp_path):
    """sample_face.jpg shows five faces, and headpose wrote one pose (the largest face's). It writes one record
    per face, each naming its face by index, and draws every head's axes and angles."""
    graph = "dms_headpose.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_sync", "--graph", "{}/{}".format(GRAPHS, graph), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.loads(report.read_text(encoding="utf-8"))["frames"][0]
    faces = frame["nodes"]["face"]["payload"]["items"]
    poses = [item["numbers"] for item in frame["nodes"]["headpose"]["payload"]["items"]]
    assert len(faces) == 5, faces
    assert [pose["face_index"] for pose in poses] == [0, 1, 2, 3, 4], poses
    assert all(-90 < pose[name] < 90 for pose in poses for name in ("yaw", "pitch", "roll")), poses


@pytest.mark.graph_e2e
def test_people_count_counts_and_decides_per_stream(tmp_path):
    """count and condition keep their state per stream: each camera has its own total, and a crowded event names
    the stream it happened in."""
    graph = "multistream_people_count.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report, events = tmp_path / "report.json", tmp_path / "events.jsonl"
    result = run("multi_model_graph_sync", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--report", str(report), "--events", str(events))
    assert result.returncode == 0, result.stdout + result.stderr
    frames = json.loads(report.read_text(encoding="utf-8"))["frames"]
    assert [f["stream"] for f in frames] == ["cam1", "cam2"]
    crowded = []
    for f in frames:
        people = [b for b in f["nodes"]["od"]["payload"]["items"] if b["class_name"] == "person"]
        total = f["nodes"]["count"]["payload"]["items"][0]["numbers"]["total"]
        value = f["nodes"]["crowded"]["payload"]["items"][0]["numbers"]["value"]
        assert total == len(people) and value == total, f["stream"]
        if total >= 5:
            crowded.append(f["stream"])
    lines = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [(e["stream"], e["node"], e["event"]) for e in lines] == [(s, "crowded", "crowded") for s in crowded]
    assert "cam2" in crowded, "sample_crowd.jpg holds more than five people"


@pytest.mark.graph_e2e
def test_hand_cascade_landmarks_sit_where_the_landmark_model_puts_them_on_the_whole_hand(tmp_path):
    """sample_hand.jpg is one hand filling the frame, so the landmark model run on the whole frame (what the
    single-model example does) is the reference. The cascade's crop must hold the whole hand: the palm detector's
    box (2x the palm) cut off the thumb tip and put the thumb tip ~230 px and the wrist ~110 px off; the edge's pad
    0.15 makes it MediaPipe's 2.6x."""
    graph = "hand_cascade.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    spec = json.loads((GRAPH_DIR / graph).read_text(encoding="utf-8"))
    whole = {"version": 1, "name": "hand_whole_frame",
             "nodes": [n for n in spec["nodes"] if n["id"] != "palm"],
             "edges": [{"from": "cam", "to": "landmark"}]}
    whole_graph = tmp_path / "whole.json"
    whole_graph.write_text(json.dumps(whole), encoding="utf-8")

    def landmarks(graph_path, name):
        report = tmp_path / (name + ".json")
        result = run("multi_model_graph_sync", "--graph", str(graph_path), "--report", str(report))
        assert result.returncode == 0, result.stdout + result.stderr
        frame = json.loads(report.read_text(encoding="utf-8"))["frames"][0]
        runs = frame.get("roi_nodes", {}).get("landmark") or [frame["nodes"]["landmark"]]
        assert len(runs) == 1, "one hand on the sample"
        a, b, tx, c, d, ty = runs[0]["origin"]["inv_align"]   # crop-local -> source pixels
        items = runs[0]["payload"]["items"]
        assert len(items) == 1 and len(items[0]["keypoints"]) == 21
        return [(a * k["x"] + b * k["y"] + tx, c * k["x"] + d * k["y"] + ty) for k in items[0]["keypoints"]]

    reference = landmarks(whole_graph, "whole")
    cascade = landmarks(GRAPH_DIR / graph, "cascade")
    # 2 % of the 3009 px frame; the hand spans about 2500 px.
    names = {0: "wrist", 4: "thumb tip", 8: "index tip", 12: "middle tip", 16: "ring tip", 20: "pinky tip"}
    for i, name in names.items():
        dx, dy = cascade[i][0] - reference[i][0], cascade[i][1] - reference[i][1]
        assert (dx * dx + dy * dy) ** 0.5 < 60, "{} is {:.0f} px off".format(name, (dx * dx + dy * dy) ** 0.5)


@pytest.mark.graph_e2e
def test_logistics_volume_measures_the_package_on_its_sample(tmp_path):
    """The segmenter's 500x333 masks and the depth model's 768x768 map both cover the frame: volume gives a median
    depth and a volume_proxy (it gave none before Phase 4)."""
    graph = "logistics_volume.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_sync", "--graph", "{}/{}".format(GRAPHS, graph), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(report.read_text(encoding="utf-8"))["frames"][0]["nodes"]["volume"]["payload"]["items"]
    assert rows, "sample_baggage_cart_airport.jpg holds a suitcase"
    for row in rows:
        n = row["numbers"]
        assert n["mask_area_px"] > 0, n
        assert 0.0 <= n["median_depth"] <= 1.0, n          # the depth map is normalized per frame
        assert n["volume_proxy"] == pytest.approx(n["mask_area_px"] * n["median_depth"]), n


@pytest.mark.graph_e2e
def test_logistics_volume_alert_sums_the_packages_and_raises_load_high(tmp_path):
    graph = "logistics_volume_alert.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report, events = tmp_path / "report.json", tmp_path / "events.jsonl"
    result = run("multi_model_graph_sync", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--report", str(report), "--events", str(events))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.loads(report.read_text(encoding="utf-8"))["frames"][0]
    rows = frame["nodes"]["volume"]["payload"]["items"]
    assert rows, "sample_baggage_cart_airport.jpg holds a suitcase"
    total = sum(r["numbers"].get("volume_proxy", 0.0) for r in rows)
    value = frame["nodes"]["load"]["payload"]["items"][0]["numbers"]["value"]
    assert value == pytest.approx(total)
    lines = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [(e["node"], e["event"]) for e in lines] == ([("load", "load_high")] if total > 100 else [])
    assert total > 100, "the sample's suitcase passes the template's threshold"


# ---------------------------------------------------------------- tier 3


@pytest.mark.graph_parity
@pytest.mark.parametrize("graph", RUNNABLE_SAMPLE_GRAPHS)
def test_sync_and_async_agree_on_real_models(tmp_path, graph):
    """The whole point of shipping both executors.

    Every runnable sample graph, compared as parsed JSON and as bytes: the
    report is the certification that ships, and `FrameReport::operator==`
    compares every shape exactly. Graphs in NPU_WEIGHT_EXCLUDED are omitted.
    """
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    sync_report = tmp_path / "sync.json"
    async_report = tmp_path / "async.json"
    a = run("multi_model_graph_sync", "--graph", "{}/{}".format(GRAPHS, graph),
            "--report", str(sync_report))
    b = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
            "--report", str(async_report))
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    with open(str(sync_report)) as fh:
        expected = json.load(fh)
    with open(str(async_report)) as fh:
        actual = json.load(fh)
    assert expected == actual
    assert sync_report.read_bytes() == async_report.read_bytes()


@pytest.mark.graph_e2e
def test_report_carries_crop_size_for_every_roi_result(tmp_path):
    """Task 12 added crop_size to RoiRef and to operator==; the report
    serializer never learned of it, so the only parity check that runs on
    hardware could not see it. shape.hpp tells consumers to PREFER it over
    src_box.size(), because an aligned crop's pixel size is not its source
    box's size."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_async", "--graph",
                 "{}/{}".format(GRAPHS, graph), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    with open(str(report), encoding="utf-8") as handle:
        frame = json.load(handle)["frames"][0]
    crops = frame["roi_nodes"]["reid"]
    assert crops, "no crops to check"
    for crop in crops:
        size = crop["origin"]["crop_size"]
        assert len(size) == 2, crop["origin"]
        assert size[0] > 0 and size[1] > 0, crop["origin"]


@pytest.mark.graph_e2e
def test_a_class_name_typo_is_reported_rather_than_silently_dropping_crops(
        tmp_path):
    """--check cannot catch this: the class list is only wrong at run
    time. Without the warning the run exits 0, says "1 frame, 0 failed",
    and hands the consumer nothing."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    for edge in spec["edges"]:
        if "roi" in edge and "classes" in edge["roi"]:
            edge["roi"]["classes"] = ["persons"]
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(spec))

    result = run("multi_model_graph_async", "--graph", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "warning:" in combined, combined
    assert "produced no crops" in combined, combined
    # The user's typo next to the truth.
    assert "persons" in combined, combined
    assert "person" in combined, combined

    # And the unmodified graph must stay quiet.
    clean = run("multi_model_graph_async", "--graph",
                "{}/{}".format(GRAPHS, graph))
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "produced no crops" not in (clean.stdout + clean.stderr)


# ------------------------------------------------------------------ streams (SP2)


@pytest.mark.graph
def test_a_bare_input_is_ambiguous_with_two_sources(tmp_path):
    result = run(BINARIES[1], "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "clip.mp4", "--model-dir", str(tmp_path))
    assert result.returncode == 2, result.stdout + result.stderr
    assert ('--input "clip.mp4" does not say which source it replaces: graph '
            '"two-cameras-reid" has 2 source nodes (cam1, cam2); write --input '
            '<source_id>=<uri>, e.g. --input cam1=clip.mp4') in result.stderr
    assert "graph:" not in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("inputs,needle", [
    (["cam1=a.mp4", "cam1=b.mp4"], '--input names source "cam1" twice'),
    (["cam2="], '--input cam2= has no uri after "="'),
])
def test_input_binding_errors_are_usage_errors(tmp_path, inputs, needle):
    args = ["--graph", "{}/{}".format(GRAPHS, MULTISTREAM), "--model-dir", str(tmp_path)]
    for value in inputs:
        args += ["--input", value]
    result = run(BINARIES[1], *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert needle in result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize("value,needle", [
    ("", "--input needs a value"),
    ("cam=", '--input cam= has no uri after "="'),
])
def test_an_empty_input_is_a_usage_error(tmp_path, binary, value, needle):
    """An empty --input never becomes an override that hides the source
    node's own "uri": refused before anything loads, with or without a name."""
    result = run(binary, "--graph", "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--input", value, "--model-dir", str(tmp_path))
    assert result.returncode == 2, result.stdout + result.stderr
    assert needle in result.stderr
    assert "graph:" not in result.stdout


@pytest.mark.graph
def test_a_live_stream_with_image_output_is_refused(tmp_path):
    """SP1's refusal, per stream: the graph's own cam1 image is fine, cam2 is live."""
    result = run(BINARIES[1], "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam2=camera:0", "--output", str(tmp_path / "out.png"),
                 "--model-dir", str(tmp_path))
    assert result.returncode == 2, result.stdout + result.stderr
    assert '"camera:0" is a live source that never ends' in result.stderr
    assert "graph:" not in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("value,uri", [
    ("cam=camera:0", "camera:0"),                            # the named form, one source
    ("rtsp://127.0.0.1:1/x?a=b", "rtsp://127.0.0.1:1/x?a=b"),  # '=' in a bare uri
])
def test_one_source_takes_a_bare_or_a_named_input(tmp_path, value, uri):
    """Review Focus #5: the live refusal names the uri the binding chose."""
    result = run(BINARIES[1], "--graph", "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--input", value, "--output", str(tmp_path / "out.png"),
                 "--model-dir", str(tmp_path))
    assert result.returncode == 2, result.stdout + result.stderr
    assert '"{}" is a live source that never ends'.format(uri) in result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("ext,first", [(".avi", "out_cam1.avi"),
                                       (".png", "out_cam1_000000.png")])
def test_per_stream_output_into_a_missing_directory_fails_before_models_load(
        tmp_path, ext, first):
    """SP1's directory check, per stream: the message names the first file a
    stream would write, not --output itself, which no stream writes."""
    target = tmp_path / "no_such_dir" / ("out" + ext)
    report = tmp_path / "report.json"
    result = run(BINARIES[1], "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--output", str(target), "--report", str(report),
                 "--model-dir", str(tmp_path))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not write {}: directory {} does not exist".format(
        target.parent / first, target.parent) in result.stderr, result.stderr
    assert "graph:" not in result.stdout  # nothing was built
    assert not report.exists()


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_help_describes_input_per_source(binary):
    result = run(binary, "--help")
    assert result.returncode == 0
    assert "--input [<source>=]<uri>" in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_help_names_the_per_stream_output_files(binary):
    result = run(binary, "--help")
    assert result.returncode == 0
    assert "<stem>_<source_id><ext>" in result.stdout
    assert "<stem>_<source_id>_<index><ext>" in result.stdout


@pytest.mark.graph_parity
def test_two_file_streams_run_in_turn_and_sync_equals_async(tmp_path):
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    cam1, cam2 = write_two_stream_videos(tmp_path)
    files = {}
    for binary, extra in (("multi_model_graph_sync", []),
                          ("multi_model_graph_async", ["--max-inflight", "4"])):
        report = tmp_path / (binary + ".json")
        result = run(binary, "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                     "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2),
                     "--report", str(report), *extra)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "stream cam1: 8 frames, 0 failed" in result.stdout
        assert "stream cam2: 5 frames, 0 failed" in result.stdout
        assert "13 frames, 0 failed" in result.stdout
        files[binary] = report.read_bytes()
    assert files["multi_model_graph_sync"] == files["multi_model_graph_async"]
    frames = json.loads(files["multi_model_graph_sync"].decode("utf-8"))["frames"]
    assert [(f["stream"], f["index"]) for f in frames] == \
        read_in_turn([("cam1", 8), ("cam2", 5)])
    for f in frames:
        assert not f["error"], f["error"]
        other = "cam2" if f["stream"] == "cam1" else "cam1"
        assert f["stream"] in f["nodes"] and other not in f["nodes"]


@pytest.mark.graph_parity
@pytest.mark.parametrize("binary", BINARIES)
def test_each_stream_equals_that_stream_run_alone(tmp_path, binary):
    """Review Focus #3: a stream's reports - track ids included - do not
    depend on the other stream's frames."""
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    cam1, cam2 = write_two_stream_videos(tmp_path)
    both = tmp_path / "both.json"
    result = run(binary, "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2),
                 "--report", str(both))
    assert result.returncode == 0, result.stdout + result.stderr
    with open(str(both), encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    for name, video in (("cam1", cam1), ("cam2", cam2)):
        alone = tmp_path / "{}.json".format(name)
        result = run(binary, "--graph", str(one_stream_graph(tmp_path, name)),
                     "--input", str(video), "--report", str(alone))
        assert result.returncode == 0, result.stdout + result.stderr
        with open(str(alone), encoding="utf-8") as handle:
            expected = json.load(handle)["frames"]
        mine = [{k: v for k, v in f.items() if k != "stream"}
                for f in frames if f["stream"] == name]
        assert mine == expected, name


@pytest.mark.graph_e2e
def test_a_camera_and_a_file_stream_end_independently(tmp_path):
    """Review Focus #5 and decision 6: --frames is per stream; the 5-frame
    file ends first and the camera carries on to 7."""
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    if not camera_available():
        pytest.skip("no camera at /dev/video0")
    _, cam2 = write_two_stream_videos(tmp_path)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam1=camera:0", "--input", "cam2={}".format(cam2),
                 "--frames", "7", "--report", str(report))
    if result.returncode != 0:
        skip_if_the_camera_did_not_open(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    with open(str(report), encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    assert [(f["stream"], f["index"]) for f in frames] == \
        read_in_turn([("cam1", 7), ("cam2", 5)])
    assert "stream cam1: 7 frames" in result.stdout
    assert "stream cam2: 5 frames" in result.stdout


@pytest.mark.graph_e2e
@pytest.mark.parametrize("ext", [".avi", ".png"])
def test_output_is_written_per_stream(tmp_path, ext):
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    cam1, cam2 = write_two_stream_videos(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / ("result" + ext)
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2),
                 "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    if ext == ".avi":
        assert sorted(p.name for p in out_dir.iterdir()) == ["result_cam1.avi", "result_cam2.avi"]
        assert "output: {}".format(out_dir / "result_cam1.avi") in result.stdout
        count1, _, shape1 = read_video(out_dir / "result_cam1.avi")
        count2, _, shape2 = read_video(out_dir / "result_cam2.avi")
        assert (count1, count2) == (8, 5)
        assert shape1 == shifted_sample_frames(1)[0].shape[:2]
        assert shape2 == shifted_sample_frames(1, image="sample_person_a1.jpg")[0].shape[:2]
    else:
        expected = (["result_cam1_{:06d}.png".format(k) for k in range(8)] +
                    ["result_cam2_{:06d}.png".format(k) for k in range(5)])
        assert sorted(p.name for p in out_dir.iterdir()) == sorted(expected)


@pytest.mark.graph_e2e
def test_display_shows_every_stream_and_leaves_the_report_alone(tmp_path):
    """One window holds every stream, a grid of labelled cells (its geometry:
    graph_engine_test's TestDisplayGridTilesTheStreams)."""
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    cam1, cam2 = write_two_stream_videos(tmp_path)
    inputs = ["--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2)]
    plain, shown = tmp_path / "plain.json", tmp_path / "shown.json"
    a = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
            "--report", str(plain), *inputs)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    b = run_with_env("multi_model_graph_async", env, "--graph",
                     "{}/{}".format(GRAPHS, MULTISTREAM), "--report", str(shown),
                     "--display", *inputs)
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    assert "13 frames, 0 failed" in b.stdout
    assert plain.read_bytes() == shown.read_bytes()


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_an_empty_stream_is_reported_after_the_other_streams_finish(tmp_path, binary):
    """Spec R9: a stream whose source yields no frame does not stop the
    others; each empty stream is named once they end, and the run exits 1
    with a complete report of every frame the other streams gave - and
    every line a finished run prints: the output video, the report and
    each stream's summary."""
    import cv2
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    cam1, _ = write_two_stream_videos(tmp_path)
    empty = tmp_path / "zero.avi"
    writer = cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    writer.release()
    report = tmp_path / "report.json"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = run(binary, "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(empty),
                 "--report", str(report), "--output", str(out_dir / "result.avi"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'ERROR [GRAPH_SCHEMA] source "{}": produced no frames'.format(empty) \
        in result.stderr, result.stderr
    assert result.stderr.count("produced no frames") == 1, result.stderr  # cam1 is not blamed
    with open(str(report), encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    assert [(f["stream"], f["index"]) for f in frames] == [("cam1", k) for k in range(8)]
    assert "output: {}".format(out_dir / "result_cam1.avi") in result.stdout, result.stdout
    assert "result_cam2" not in result.stdout, result.stdout  # it wrote nothing
    assert read_video(out_dir / "result_cam1.avi")[0] == 8
    assert "report: {}".format(report) in result.stdout, result.stdout
    assert "stream cam1: 8 frames, 0 failed" in result.stdout, result.stdout
    assert "stream cam2: 0 frames, 0 failed" in result.stdout, result.stdout
    assert "8 frames, 0 failed" in result.stdout.splitlines(), result.stdout


@pytest.mark.graph_e2e
def test_each_stream_has_its_own_input_line():
    """Spec §2.5: with several sources the loop names every stream's input,
    in declaration order, as "input:  <source>: <description>"."""
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, MULTISTREAM))
    assert result.returncode == 0, result.stdout + result.stderr
    inputs = [line for line in result.stdout.splitlines() if line.startswith("input:")]
    assert inputs == ["input:  cam1: Image file: sample/img/sample_people.jpg",
                      "input:  cam2: Image file: sample/img/sample_person_a1.jpg"], \
        result.stdout


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_a_stop_request_ends_every_stream_with_readable_videos(tmp_path, binary):
    """SIGTERM mid-run with two streams: both stop reading, the frames in
    flight are finished, each stream's .mkv is finalized and holds exactly
    that stream's reported frames, and the report is complete JSON."""
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    total = 1500
    cam1 = write_long_video(tmp_path / "long1.avi", total)
    cam2 = write_long_video(tmp_path / "long2.avi", total)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    report = tmp_path / "report.json"
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                       "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2),
                       "--output", str(out_dir / "result.mkv"), "--report", str(report)])
    try:
        run_.wait_looping()
        time.sleep(1.0)  # some frames into the loop
        run_.stop(signal.SIGTERM)
    finally:
        run_.close()
    assert run_.proc.returncode == 0, run_.output()
    stdout = run_.stdout()
    with open(str(report), encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]  # valid JSON: the file was finished
    counts = {}
    for name in ("cam1", "cam2"):
        indices = [f["index"] for f in frames if f["stream"] == name]
        assert 0 < len(indices) < total, (name, len(indices))
        assert indices == list(range(len(indices))), name
        counts[name] = len(indices)
        video = out_dir / "result_{}.mkv".format(name)
        assert "output: {}".format(video) in stdout, run_.output()
        assert read_video(video)[0] == counts[name], name
        assert "stream {}: {}, 0 failed".format(name, frames_word(counts[name])) in \
            stdout.splitlines(), run_.output()
    assert "{}, 0 failed".format(frames_word(len(frames))) in stdout.splitlines(), run_.output()


def frames_word(n):
    """The CLI's "<n> frame(s)"."""
    return "{} frame{}".format(n, "" if n == 1 else "s")


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_a_stop_during_model_load_blames_no_stream(tmp_path, binary):
    """SP1's "interrupted before the first frame", with several streams: a
    stop that comes while the models load is no stream's fault."""
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    report = tmp_path / "report.json"
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                       "--report", str(report)])
    try:
        deadline = time.monotonic() + 60
        while not sigterm_is_caught(run_.proc.pid):
            assert run_.proc.poll() is None, run_.output()
            assert time.monotonic() < deadline, "no SIGTERM handler:\n" + run_.output()
            time.sleep(0.001)
        run_.stop(signal.SIGTERM, timeout=120)
    finally:
        run_.close()
    assert run_.proc.returncode == 0, run_.output()
    assert "interrupted before the first frame" in run_.stderr(), run_.output()
    assert "produced no frames" not in run_.stderr(), run_.output()
    assert "0 frames, 0 failed" in run_.stdout(), run_.output()
    with open(str(report), encoding="utf-8") as handle:
        assert json.load(handle) == {"frames": [], "graph": "two-cameras-reid"}


VITPOSE_GRAPH = {
    "version": 1, "name": "top-down",
    "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
              {"id": "od", "model": "yolov8n"}, {"id": "pose", "model": "vitpose-s_256x192"}],
    "edges": [{"from": "cam", "to": "od"},
              {"from": "od", "to": "pose", "roi": {"classes": ["person"], "max": 4}}]}


@pytest.mark.graph_parity
def test_top_down_pose_cascade_restores_keypoints_inside_each_crop(tmp_path):
    if not (MODEL_DIR / "vitpose-s_256x192.dxnn").is_file():
        pytest.skip("vitpose-s_256x192 not downloaded")
    graph = tmp_path / "vitpose.json"
    graph.write_text(json.dumps(VITPOSE_GRAPH), encoding="utf-8")
    reports = {}
    for binary in BINARIES:
        path = tmp_path / (binary + ".json")
        result = run(binary, "--graph", str(graph), "--report", str(path))
        assert result.returncode == 0, result.stdout + result.stderr
        reports[binary] = path.read_bytes()
    assert reports[BINARIES[0]] == reports[BINARIES[1]]
    crops = json.loads(reports[BINARIES[0]])["frames"][0]["roi_nodes"]["pose"]
    assert 1 <= len(crops) <= 4
    for crop in crops:
        x, y, w, h = crop["origin"]["src_box"]
        m = crop["origin"]["inv_align"]
        points = crop["payload"]["items"][0]["keypoints"]
        assert len(points) == 17
        for p in points:
            sx, sy = m[0] * p["x"] + m[1] * p["y"] + m[2], m[3] * p["x"] + m[4] * p["y"] + m[5]
            assert x - 1 <= sx <= x + w + 1 and y - 1 <= sy <= y + h + 1


# Payload-edge chains: a model's output image feeds the detector, and the
# detector's boxes feed ROI stages. Every coordinate a report carries must
# come back to the source frame through its `origin.inv_align`.
ENHANCE_CHAIN = "chain_zerodce_od_pose_emb.json"
SR_CHAIN = "chain_sr_od_pose.json"


def to_source(inv_align, x, y):
    """A point of the image a stage ran on (or of its crop), in source-frame
    coordinates."""
    m = inv_align
    return m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]


def box_to_source(inv_align, box):
    """An axis-aligned [x, y, w, h] box, in source-frame coordinates."""
    x0, y0 = to_source(inv_align, box[0], box[1])
    x1, y1 = to_source(inv_align, box[0] + box[2], box[1] + box[3])
    return [x0, y0, x1 - x0, y1 - y0]


def box_inside(box, width, height):
    x, y, w, h = box
    return 0 <= x and 0 <= y and x + w <= width and y + h <= height


def point_near_box(x, y, box, margin):
    bx, by, bw, bh = box
    return bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def model_output_width(model):
    """The width of a model's output tensor, as the runtime reads it from
    the .dxnn. A subprocess: some conftests replace dx_engine with a Mock."""
    dxnn = MODEL_DIR / _registry()[model]["dxnn_file"]
    code = ("import sys\nfrom dx_engine import InferenceEngine\n"
            "shape = InferenceEngine(sys.argv[1]).get_output_tensors_info()[0]['shape']\n"
            "width = 1\nfor n in shape[1:]:\n    width *= n\nprint(width)")
    done = run_bounded([sys.executable, "-c", code, str(dxnn)], capture_output=True,
                       text=True, cwd=str(PROJECT_ROOT), timeout=RUN_TIMEOUT_S)
    assert done.returncode == 0, done.stdout + done.stderr
    return int(done.stdout.split()[-1])


def sync_equals_async(tmp_path, graph):
    """Frame 0 of a shipped graph's --report, once both executors have
    written it byte for byte the same."""
    reports = []
    for binary in BINARIES:
        path = tmp_path / (binary + ".json")
        result = run(binary, "--graph", "{}/{}".format(GRAPHS, graph), "--report", str(path))
        assert result.returncode == 0, result.stdout + result.stderr
        reports.append(path.read_bytes())
    assert reports[0] == reports[1], graph
    return json.loads(reports[0].decode("utf-8"))["frames"][0]


@pytest.mark.graph_parity
def test_enhance_detect_roi_chain_keeps_source_coordinates(tmp_path):
    """cam -> enh (image hand-off) -> od -> {kp, emb} (ROI). The detector
    sees the enhanced image and the ROI stages see crops of it, yet every
    box, crop and keypoint must land on the cam frame."""
    absent = missing_artifacts(ENHANCE_CHAIN)
    if absent:
        pytest.skip(absent)
    frame = sync_equals_async(tmp_path, ENHANCE_CHAIN)
    cam = frame["nodes"]["cam"]["payload"]["image"]
    width, height = cam["cols"], cam["rows"]

    od = frame["nodes"]["od"]
    assert any(item["class_name"] == "person" for item in od["payload"]["items"]), od
    for item in od["payload"]["items"]:
        box = box_to_source(od["origin"]["inv_align"], item["box"])
        assert box_inside(box, width, height), (item, box, width, height)

    crops = frame["roi_nodes"]
    assert crops.get("kp") and crops.get("emb"), sorted(crops)
    for node in ("kp", "emb"):
        for crop in crops[node]:
            assert box_inside(crop["origin"]["src_box"], width, height), \
                (node, crop["origin"], width, height)

    # Which keypoints are confident depends on the model build: on the dark
    # crop this chain makes, one store's vitpose-s tops out at 0.43 and
    # another's at 0.24. So each item's top_k most confident keypoints are
    # checked whatever their score, plus every keypoint above 0.3, and each
    # must land on its crop's source box.
    top_k = 5
    checked = 0
    for crop in crops["kp"]:
        origin = crop["origin"]
        for item in crop["payload"]["items"]:
            points = sorted(item["keypoints"], key=lambda p: -p["confidence"])
            assert points, ("no keypoints in a kp item", origin)
            for rank, point in enumerate(points):
                if rank >= top_k and point["confidence"] <= 0.3:
                    break
                x, y = to_source(origin["inv_align"], point["x"], point["y"])
                assert point_near_box(x, y, origin["src_box"], 2), (point, x, y, origin)
                checked += 1
    assert checked >= top_k, "fewer than {} keypoints to check".format(top_k)

    spec = json.loads((GRAPH_DIR / ENHANCE_CHAIN).read_text(encoding="utf-8"))
    emb_model = next(node["model"] for node in spec["nodes"] if node["id"] == "emb")
    width_of_model = model_output_width(emb_model)
    for crop in crops["emb"]:
        assert len(crop["payload"]["values"]) == width_of_model, crop["origin"]


@pytest.mark.graph_e2e
@pytest.mark.skip(reason="chain_sr_od_pose.json exceeds DX-M1 NPU weight memory")
def test_sr_chain_boxes_map_back_to_the_frame(tmp_path):
    """cam -> sr (x2 hand-off) -> od -> roi -> kp. The detector runs on the
    doubled image; mapped back, its top-3 persons must be the boxes the same
    detector finds on the raw frame."""
    absent = missing_artifacts(SR_CHAIN)
    if absent:
        pytest.skip(absent)
    spec = json.loads((GRAPH_DIR / SR_CHAIN).read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in spec["nodes"]}
    raw_graph = tmp_path / "raw_od.json"
    raw_graph.write_text(json.dumps({
        "version": 1, "name": "raw-od", "nodes": [nodes["cam"], nodes["od"]],
        "edges": [{"from": "cam", "to": "od"}]}), encoding="utf-8")

    def persons(graph, report):
        result = run(BINARIES[0], "--graph", str(graph), "--report", str(report))
        assert result.returncode == 0, result.stdout + result.stderr
        od = json.loads(report.read_text(encoding="utf-8"))["frames"][0]["nodes"]["od"]
        items = [item for item in od["payload"]["items"] if item["class_name"] == "person"]
        items.sort(key=lambda item: -item["score"])
        return [box_to_source(od["origin"]["inv_align"], item["box"]) for item in items]

    on_sr = persons("{}/{}".format(GRAPHS, SR_CHAIN), tmp_path / "sr.json")[:3]
    on_raw = persons(raw_graph, tmp_path / "raw.json")
    assert on_sr and on_raw, (on_sr, on_raw)
    for box in on_sr:
        best = max(iou(box, raw) for raw in on_raw)
        assert best >= 0.5, (box, on_raw, best)


@pytest.mark.graph_parity
@pytest.mark.parametrize("model,dxnn,factor", [("espcn-x2_17x17", "espcn-x2_17x17.dxnn", 2),
                                               ("espcn-x4_17x17", "espcn-x4_17x17.dxnn", 4)])
def test_espcn_in_a_graph_equals_the_runner_output(tmp_path, model, dxnn, factor):
    import cv2
    import numpy
    if not (MODEL_DIR / dxnn).is_file():
        pytest.skip("{} not downloaded".format(model))
    runner = resolve_bin_dir() / (model + "_sync")
    if not runner.exists():
        pytest.skip("{} not built".format(runner.name))
    image = PROJECT_ROOT / "sample" / "img" / "sample_lowres275x150.png"
    save = tmp_path / "sr.png"
    env = dict(os.environ, DXAPP_SAVE_IMAGE=str(save))
    done = run_bounded([str(runner), "-m", str(MODEL_DIR / dxnn), "-i", str(image), "--no-display"],
                       capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env,
                       timeout=RUN_TIMEOUT_S)
    assert done.returncode == 0, done.stdout + done.stderr
    expected = cv2.imread(str(tmp_path / "sr_output_only.png"))
    assert expected is not None and expected.shape == (150 * factor, 275 * factor, 3)
    graph = tmp_path / "sr.json"
    graph.write_text(json.dumps({
        "version": 1, "nodes": [{"id": "cam", "type": "source", "uri": str(image)},
                                {"id": "sr", "model": model}],
        "edges": [{"from": "cam", "to": "sr"}]}), encoding="utf-8")
    report = tmp_path / "r.json"
    result = run("multi_model_graph_sync", "--graph", str(graph), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.load(open(str(report), encoding="utf-8"))["frames"][0]["nodes"]["sr"]["payload"]["image"]
    assert (summary["rows"], summary["cols"]) == expected.shape[:2]
    from test_graph_python import load_dx_graph
    dx_graph = load_dx_graph()
    with dx_graph.Graph(str(graph)) as g:
        d = g.run(cv2.imread(str(image))).to_dict()
    assert numpy.array_equal(d["nodes"]["sr"]["payload"]["image_array"], expected)


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_a_one_source_run_on_an_empty_video_exits_1_with_an_empty_report(tmp_path, binary):
    """SP1 leave item: a source that gives no frame at all is an error
    (exit 1, "produced no frames" on stderr), and --report is still a
    finished, empty report."""
    import cv2
    graph = "cascade_od_attr.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    empty = tmp_path / "zero.avi"
    writer = cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    writer.release()
    report = tmp_path / "report.json"
    result = run(binary, "--graph", "{}/{}".format(GRAPHS, graph),
                 "--input", str(empty), "--report", str(report))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "produced no frames" in result.stderr, result.stderr
    with open(str(report), encoding="utf-8") as handle:
        assert json.load(handle) == {"frames": [], "graph": "person-attributes"}


@pytest.mark.graph
def test_check_widens_its_columns_for_long_names(tmp_path):
    """SP2 leave item: an id of 10+ or a model name of 30+ characters ran
    into the next column. A space now always follows; short names print as
    before (the other --check tests pin that)."""
    spec = {"version": 1, "name": "wide",
            "nodes": [{"id": "front_door_camera", "type": "source",
                       "uri": "sample/img/sample_face.jpg"},
                      {"id": "face_detector_1", "model": "retinaface_mobilenetv1_736x1280"}],
            "edges": [{"from": "front_door_camera", "to": "face_detector_1"}]}
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(spec))
    # Columns, not artifacts: an empty model dir keeps a broken file in
    # assets/models from turning this into a MODEL_LOAD exit.
    no_models = tmp_path / "no_models"
    no_models.mkdir()
    result = run(BINARIES[0], "--check", str(path), "--model-dir", str(no_models))
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line for line in result.stdout.splitlines() if line.startswith("face_detector_1")]
    assert rows and re.match(r"face_detector_1 +retinaface_mobilenetv1_736x1280 "
                             r"+boxes +full_frame +", rows[0]), result.stdout
    assert re.search(r"^front_door_camera +\(source\) +frame", result.stdout, re.M), result.stdout


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_a_later_sigint_ends_a_run_stuck_writing_its_output(tmp_path, binary):
    """U-36: --output is a FIFO nobody reads, so the CLI hangs in open()
    after the frame. The first SIGINT is a graceful request and cannot
    finish; a second one more than 200 ms later ends the process."""
    graph = "cascade_od_attr.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    fifo = tmp_path / "stuck.png"
    os.mkfifo(str(fifo))
    run_ = LoopingRun([str(path), "--graph", "{}/{}".format(GRAPHS, graph),
                       "--output", str(fifo)])
    try:
        run_.wait_looping()
        # The image through both models; now blocked opening the FIFO.
        wait_until_blocked_opening_a_fifo(run_.proc, output=run_.output)
        assert run_.proc.poll() is None, "finished before the first SIGINT:\n" + run_.output()
        run_.proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        assert run_.proc.poll() is None, "one SIGINT ended a run stuck in open():\n" + run_.output()
        run_.proc.send_signal(signal.SIGINT)
        run_.proc.wait(timeout=10)
    finally:
        run_.close()
    assert run_.proc.returncode == -signal.SIGINT, run_.output()


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_a_model_without_config_json_runs_on_its_defaults_silently(binary, tmp_path):
    """U-75: realesrgan_x2 has no config.json; the graph used to print
    "Config file not found" for it (and 75 other models).

    A one-node graph: handoff_sr_od_cls.json also loads yolov8-n and resnet50,
    which together exceed DX-M1 weight memory.
    """
    dxnn = MODEL_DIR / "realesrgan-x2_192x192.dxnn"
    if not dxnn.is_file():
        pytest.skip("{} not downloaded".format(dxnn.name))
    image = PROJECT_ROOT / "sample" / "img" / "sample_lowres165x90.png"
    graph = tmp_path / "sr_only.json"
    graph.write_text(json.dumps({
        "version": 1,
        "nodes": [{"id": "cam", "type": "source", "uri": str(image)},
                  {"id": "sr", "model": "realesrgan_x2"}],
        "edges": [{"from": "cam", "to": "sr"}],
    }), encoding="utf-8")
    result = run(binary, "--graph", str(graph))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Config file not found" not in result.stdout + result.stderr


# ── The per-variant registry (release port, R6/R9/R12) ─────────────────────


def _list_rows(stdout):
    """{first token: line} for every row --list-models prints."""
    return {line.split()[0]: line for line in stdout.splitlines()
            if line.strip() and not line.startswith("-") and not line.startswith("model ")}


def _registry_rows():
    with open(str(PROJECT_ROOT / "config" / "model_registry.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def dxrt_version():
    """(major, minor, patch) from ``dxrt-cli --version`` ("DXRT v3.4.1+baec914"),
    or None when there is no dxrt-cli or it prints no version."""
    try:
        out = subprocess.run(["dxrt-cli", "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"DXRT v?(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(part) for part in match.groups()) if match else None


# A real q-lite-dxnn/2_5_0 (container v9) espcn-x2_17x17.dxnn, when a caller
# has one: the directory that holds it. Without it the test writes a v9
# header stub, which is all the check reads.
V9_MODEL_DIR = os.environ.get("DXAPP_V9_MODEL_DIR", "")


def v9_model_dir(tmp_path, model):
    """(directory, file name) of a container-v9 .dxnn for `model`."""
    dxnn = _registry()[model]["dxnn_file"]
    if V9_MODEL_DIR and (Path(V9_MODEL_DIR) / dxnn).is_file():
        return Path(V9_MODEL_DIR), dxnn
    directory = tmp_path / "v9"
    directory.mkdir(exist_ok=True)
    (directory / dxnn).write_bytes(b"DXNN" + (9).to_bytes(4, "little") + b"{}")
    return directory, dxnn


def one_node_graph(tmp_path, model):
    path = tmp_path / "one_node.json"
    path.write_text(json.dumps({
        "version": 1, "name": "one-node",
        "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
                  {"id": "m", "model": model}],
        "edges": [{"from": "cam", "to": "m"}]}))
    return path


def require_an_older_runtime():
    version = dxrt_version()
    if version is None or version >= (3, 5, 0):
        pytest.skip("needs DX-RT older than 3.5.0 (dxrt-cli --version: {})".format(version))


# The tiled-SR maker (espcn-x2: one input channel) and a typed stage.
V9_MODELS = ["espcn-x2_17x17", "yolov8-n_640x640"]


@pytest.mark.graph
@pytest.mark.parametrize("model", V9_MODELS)
def test_v9_model_on_an_older_runtime_fails_before_loading(tmp_path, model):
    """R12 / spec section 4: a container-v9 .dxnn on DX-RT < 3.5.0 is refused
    by the graph paths, naming the file, before any engine is created -
    --check exits 1, and a run fails with the same text before it builds the
    graph (no "graph:" line) and before dxrt says anything."""
    require_an_older_runtime()
    model_dir, dxnn = v9_model_dir(tmp_path, model)
    graph = one_node_graph(tmp_path, model)

    for binary in BINARIES:
        check = run(binary, "--graph", str(graph), "--check", "--model-dir", str(model_dir))
        assert check.returncode == 1, check.stdout + check.stderr
        assert "needs DX-RT >= 3.5.0" in check.stderr, check.stderr
        assert str(model_dir / dxnn) + ": .dxnn container v9 needs DX-RT >= 3.5.0" in check.stderr
        assert "Use the v8 file (dxnn/2_4_0) or upgrade DX-RT." in check.stderr
        assert "{}  [present, v9 (needs DX-RT >= 3.5.0)]".format(dxnn) in check.stdout, check.stdout

        result = run(binary, "--graph", str(graph), "--model-dir", str(model_dir))
        assert result.returncode == 1, result.stdout + result.stderr
        assert "graph:" not in result.stdout, result.stdout
        assert "dxrt-exception" not in result.stderr, result.stderr  # dxrt never ran
        first = result.stderr.splitlines()[0] if result.stderr else ""
        assert first.startswith("ERROR [MODEL_LOAD] node \"m\""), result.stderr
        assert (str(model_dir / dxnn) + ": .dxnn container v9 needs DX-RT >= 3.5.0, but this "
                "runtime is ") in first, result.stderr
        # I3: the hint names the v8 file; downloading again fetches this v9 file.
        assert "  -> use the v8 file (dxnn/2_4_0) or upgrade DX-RT to >= 3.5.0" in result.stderr
        assert "setup.sh" not in result.stderr, result.stderr
        assert "download_models" not in result.stderr, result.stderr
        # --check says what to do too, the same hint as the run.
        assert "  -> use the v8 file (dxnn/2_4_0) or upgrade DX-RT to >= 3.5.0" in check.stderr


@pytest.mark.graph
def test_list_models_shows_published_and_container_version(tmp_path):
    """--list-models prints the registry's published flag and, per row, the
    container version of the .dxnn in --model-dir (bytes 4-7 after DXNN),
    or "missing". The store's yolov8-n_640x640 must read as the version its
    own header carries (v8 or v9 store alike); a stand-in DXNN+9 file reads
    as v9."""
    header = run(BINARIES[0], "--list-models", "--model-dir", str(MODEL_DIR))
    assert header.returncode == 0, header.stderr
    assert header.stdout.splitlines()[0].split() == [
        "model", "task", "produces", "consumes", "input", "published", "file", "ready"]
    if not (MODEL_DIR / "yolov8-n_640x640.dxnn").is_file():
        pytest.skip("{} has no yolov8-n_640x640.dxnn".format(MODEL_DIR))
    with open(MODEL_DIR / "yolov8-n_640x640.dxnn", "rb") as dxnn:
        magic, version = dxnn.read(4), dxnn.read(4)
    assert magic == b"DXNN", magic
    expected = "v{}".format(int.from_bytes(version, "little"))
    row = _list_rows(header.stdout)["yolov8-n_640x640"].split()
    assert row[5:8] == ["yes", expected, "yes"], (row, expected)

    (tmp_path / "yolov8-n_640x640.dxnn").write_bytes(b"DXNN" + (9).to_bytes(4, "little") + b"{}")
    (tmp_path / "resnet50_224x224.dxnn").write_bytes(b"ONNX-not-a-dxnn")
    result = run(BINARIES[0], "--list-models", "--model-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    rows = _list_rows(result.stdout)
    assert rows["yolov8-n_640x640"].split()[6] == "v9"
    older = dxrt_version()
    if older is not None and older < (3, 5, 0):
        # R12: a v9 file this runtime cannot load says so in the same column.
        assert "v9 (needs DX-RT >= 3.5.0)" in rows["yolov8-n_640x640"], rows["yolov8-n_640x640"]
        assert "needs DX-RT" not in rows["resnet50_224x224"]
    assert rows["resnet50_224x224"].split()[6] == "invalid"
    assert rows["yolov8-s-pose_640x640"].split()[6] == "missing"
    published = {e["variant"]: e.get("published") for e in _registry_rows()}
    unpublished = sorted(v for v, p in published.items() if not p)
    for variant in unpublished[:3]:
        assert rows[variant].split()[5] == "no", rows[variant]


@pytest.mark.graph
def test_list_models_counts_every_variant_and_alias():
    """One row per variant plus one per alias_of row (R6), both counted from
    the registry; deit_base384_distilled is listed as an alias of
    deit-b_384x384_distilled. Old model_names are aliases too, but are not listed."""
    entries = _registry_rows()
    variants = {e["variant"] for e in entries}
    alias_of = {e["model_name"]: e["alias_of"] for e in entries if e.get("alias_of")}
    by_name = {e["model_name"]: e["variant"] for e in entries}
    result = run(BINARIES[0], "--list-models")
    assert result.returncode == 0, result.stderr
    rows = _list_rows(result.stdout)
    assert len(rows) == len(variants) + len(alias_of), (len(rows), len(variants), len(alias_of))
    assert set(rows) == variants | set(alias_of)
    for name, target in alias_of.items():
        assert rows[name].split()[1:] == ["alias", "of", by_name[target]], rows[name]
    assert rows["deit_base384_distilled"].split() == [
        "deit_base384_distilled", "alias", "of", "deit-b_384x384_distilled"]
    assert "yolov8n" not in rows


def _write_graph(tmp_path, nodes, edges=None):
    graph = {"version": 1, "name": "port",
             "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"}] + nodes,
             "edges": edges if edges is not None else [
                 {"from": "cam", "to": node["id"]} for node in nodes]}
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


@pytest.mark.graph
def test_check_notes_old_names_and_aliases(tmp_path):
    """R6: an old name and an alias_of name both resolve, with a note; the
    node table names the variant that runs."""
    path = _write_graph(tmp_path, [{"id": "od", "model": "yolov8n"},
                                   {"id": "cls", "model": "deit_base384_distilled"}])
    result = run(BINARIES[0], "--check", str(path), "--model-dir", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'note: node "od": "yolov8n" is the old name of "yolov8-n_640x640"' in result.stdout
    assert ('note: node "cls": "deit_base384_distilled" is an alias of "deit-b_384x384_distilled"'
            in result.stdout), result.stdout
    table = {line.split()[0]: line.split() for line in result.stdout.splitlines() if line.strip()}
    assert table["od"][1] == "yolov8-n_640x640"
    assert table["cls"][1] == "deit-b_384x384_distilled"


# The notes --check prints for an old name and an alias_of name (R6).
ALIAS_NOTES = ['note: node "od": "yolov8n" is the old name of "yolov8-n_640x640"',
               'note: node "cls": "deit_base384_distilled" is an alias of "deit-b_384x384_distilled"']


@pytest.mark.graph
def test_a_run_notes_old_names_and_aliases_on_stderr(tmp_path):
    """C5 deit ruling: a run never resolves a name silently. It prints the
    note --check prints, on stderr, once per aliased node, before anything
    else fails (here: the empty --model-dir), in both executors."""
    path = _write_graph(tmp_path, [{"id": "od", "model": "yolov8n"},
                                   {"id": "cls", "model": "deit_base384_distilled"}])
    empty = tmp_path / "models"
    empty.mkdir()
    for binary in BINARIES:
        result = run(binary, "--graph", str(path), "--model-dir", str(empty))
        assert result.returncode == 1, result.stdout + result.stderr
        lines = result.stderr.splitlines()
        assert lines[:2] == ALIAS_NOTES, result.stderr
        assert lines[2].startswith("ERROR [MODEL_MISSING]"), result.stderr
        for note in ALIAS_NOTES:
            assert result.stderr.count(note) == 1, result.stderr
            assert note not in result.stdout


@pytest.mark.graph
def test_a_run_by_an_old_name_notes_it_once_and_reports_as_the_variant(tmp_path):
    """The note is stderr only: a graph naming "yolov8n" writes the report its
    variant name writes, byte for byte, with one note on stderr; the variant
    name prints none."""
    if not (MODEL_DIR / "yolov8-n_640x640.dxnn").is_file():
        pytest.skip("{} has no yolov8-n_640x640.dxnn".format(MODEL_DIR))
    for binary in BINARIES:
        reports = {}
        for model in ("yolov8n", "yolov8-n_640x640"):
            graph = tmp_path / "{}.json".format(model)
            graph.write_text(json.dumps({
                "version": 1, "name": "port",
                "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
                          {"id": "od", "model": model}],
                "edges": [{"from": "cam", "to": "od"}]}), encoding="utf-8")
            report = tmp_path / "{}_{}.json".format(model, binary)
            result = run(binary, "--graph", str(graph), "--report", str(report))
            assert result.returncode == 0, result.stdout + result.stderr
            notes = [line for line in result.stderr.splitlines() if line.startswith("note: ")]
            assert notes == ([ALIAS_NOTES[0]] if model == "yolov8n" else []), result.stderr
            assert "note: " not in result.stdout
            reports[model] = report.read_bytes()
        assert reports["yolov8n"] == reports["yolov8-n_640x640"]


@pytest.mark.graph
def test_check_fails_on_an_unknown_model_name(tmp_path):
    """An unknown name is not substituted: --check exits 1 and names the node
    and the nearest variant as a hint only."""
    path = _write_graph(tmp_path, [{"id": "od", "model": "yolov8-n_640x64"}])
    result = run(BINARIES[0], "--check", str(path), "--model-dir", str(tmp_path))
    assert result.returncode == 1, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "ERROR [MODEL_UNKNOWN]" in combined and 'node "od"' in combined, combined
    assert 'did you mean "yolov8-n_640x640"?' in combined, combined


@pytest.mark.graph
def test_check_prints_the_resources_a_node_needs(tmp_path):
    """R9: a gallery file is checked for its DXGAL1 magic; a zero-shot CLIP
    node says the prompt bank is Python-only."""
    zero_shot = sorted(e["variant"] for e in _registry_rows()
                       if e["task"] == "zero_shot_image_classification")[0]
    path = _write_graph(tmp_path, [{"id": "vpr", "model": "eigenplaces-resnet18_512x512"},
                                   {"id": "clip", "model": zero_shot}])
    result = run(BINARIES[0], "--check", str(path), "--model-dir", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    gallery = "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"
    assert 'resource: node "vpr": gallery {} [present, DXGAL1]'.format(gallery) in result.stdout, \
        result.stdout
    assert 'note: node "clip": the CLIP prompt bank is Python-only' in result.stdout, result.stdout


@pytest.mark.graph
def test_check_reads_a_node_gallery_as_the_run_does(tmp_path):
    """I18: a node's "gallery" param is the gallery --check reports, read
    as the stage overlay reads it (GalleryAgainstRoot): an absolute path as
    given, a relative one against the repository."""
    repo_gallery = "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"
    copied = tmp_path / "copied_gallery.bin"
    shutil.copyfile(PROJECT_ROOT / repo_gallery, copied)
    gone = tmp_path / "no_such_gallery.bin"
    path = _write_graph(tmp_path, [
        {"id": "abs", "model": "eigenplaces-resnet18_512x512", "params": {"gallery": str(copied)}},
        {"id": "rel", "model": "eigenplaces-resnet18_512x512", "params": {"gallery": repo_gallery}},
        {"id": "gone", "model": "eigenplaces-resnet18_512x512", "params": {"gallery": str(gone)}}])
    for binary in BINARIES:
        result = run(binary, "--check", str(path), "--model-dir", str(tmp_path))
        assert result.returncode == 0, result.stdout + result.stderr
        lines = [line for line in result.stdout.splitlines() if line.startswith("resource: ")]
        assert lines == [
            'resource: node "abs": gallery {} [present, DXGAL1]'.format(copied),
            'resource: node "rel": gallery {} [present, DXGAL1]'.format(repo_gallery),
            'resource: node "gone": gallery {} [MISSING]'.format(gone),
        ], result.stdout


@pytest.mark.graph
def test_check_names_the_container_version_of_a_loadable_file(tmp_path):
    """N3: the file column says which container a present .dxnn is, also
    when this runtime loads it ("[present, v8]"), as --list-models does."""
    (tmp_path / "yolov8-n_640x640.dxnn").write_bytes(b"DXNN" + (8).to_bytes(4, "little") + b"{}")
    graph = one_node_graph(tmp_path, "yolov8-n_640x640")
    for binary in BINARIES:
        check = run(binary, "--graph", str(graph), "--check", "--model-dir", str(tmp_path))
        assert check.returncode == 0, check.stdout + check.stderr
        assert "yolov8-n_640x640.dxnn  [present, v8]" in check.stdout, check.stdout

    version = dxrt_version()
    if version is None or version < (3, 5, 0):
        return  # v9 there is the refusal, pinned by the R12 test above
    v9_dir = tmp_path / "v9"
    v9_dir.mkdir()
    (v9_dir / "yolov8-n_640x640.dxnn").write_bytes(b"DXNN" + (9).to_bytes(4, "little") + b"{}")
    for binary in BINARIES:
        check = run(binary, "--graph", str(graph), "--check", "--model-dir", str(v9_dir))
        assert check.returncode == 0, check.stdout + check.stderr
        assert "yolov8-n_640x640.dxnn  [present, v9]" in check.stdout, check.stdout


# ---------------------------------------------------------------- graph composer P1


def run_env(binary, env, *args):
    """run() with extra environment variables."""
    path = resolve_bin_dir() / binary
    if not path.exists():
        pytest.skip("{} not built".format(binary))
    return run_bounded([str(path)] + list(args), capture_output=True, text=True,
                       cwd=str(PROJECT_ROOT), timeout=RUN_TIMEOUT_S,
                       env=dict(os.environ, **env))


def _progress_lines(stdout):
    return [line for line in stdout.splitlines() if line.startswith("[PROGRESS] ")]


def _field(line, key):
    for token in line.split():
        if token.startswith(key + "="):
            return token[len(key) + 1:]
    return None


def _node_ms(line):
    value = _field(line, "node_ms")
    if value is None:
        return {}
    return {name: float(ms) for name, ms in (part.rsplit(":", 1) for part in value.split(","))}


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_help_lists_report_timing(binary):
    result = run(binary, "--help")
    assert result.returncode == 0
    assert "--report-timing" in result.stdout


@pytest.mark.graph
def test_report_timing_needs_a_report():
    result = run(BINARIES[0], "--graph", "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--report-timing")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--report-timing needs --report <file>" in result.stderr, result.stderr


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_progress_lines_carry_time_per_node(tmp_path, binary):
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    video = write_moving_video(tmp_path / "moving.avi")
    result = run_env(binary, {"DXAPP_PROGRESS": "1"}, "--graph",
                     "{}/{}".format(GRAPHS, graph), "--input", str(video))
    assert result.returncode == 0, result.stdout + result.stderr
    lines = _progress_lines(result.stdout)
    assert lines, result.stdout
    last = lines[-1]
    assert _field(last, "frames") == "8", last
    assert _field(last, "avg_fps") is not None, last
    # 8 frames fit under --max-inflight, so the async CLI returns every report in
    # one burst: the rate counts from the first submit, not across the burst.
    assert float(_field(last, "avg_fps")) < 1000, last
    times = _node_ms(last)
    assert set(times) == {"od", "reid"}, last
    assert all(ms > 0.0 for ms in times.values()), last
    assert "read_ms=" not in last, last
    # The run averages come before the summary line. (DX-RT may print its own
    # "[DXRT] ..." line at exit, so the summary is not always the last line.)
    out = result.stdout.splitlines()
    assert "8 frames, 0 failed" in out, result.stdout[-300:]
    assert out.index(last) < out.index("8 frames, 0 failed"), result.stdout[-300:]


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_report_timing_adds_node_ms_and_nothing_else(tmp_path, binary):
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    plain, timed = tmp_path / "plain.json", tmp_path / "timed.json"
    a = run(binary, "--graph", "{}/{}".format(GRAPHS, graph), "--report", str(plain))
    b = run(binary, "--graph", "{}/{}".format(GRAPHS, graph), "--report", str(timed),
            "--report-timing")
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    plain_frames = json.loads(plain.read_text())["frames"]
    timed_frames = json.loads(timed.read_text())["frames"]
    assert all("node_ms" not in frame for frame in plain_frames)
    for frame in timed_frames:
        assert set(frame["node_ms"]) == {"od", "reid"}, frame["node_ms"]
        assert frame["node_ms"]["od"] > 0.0
    assert [{k: v for k, v in f.items() if k != "node_ms"} for f in timed_frames] == plain_frames


@pytest.mark.graph
@pytest.mark.parametrize("content", [b"", b"ONNX-not-a-dxnn"], ids=["empty", "not-dxnn"])
def test_check_fails_on_a_model_file_that_is_not_a_dxnn(tmp_path, content):
    """A 0-byte or foreign file used to print [present, invalid] and exit 0;
    a run then failed inside DX-RT. Like a v9 file on an old runtime, it is a
    graph that cannot run with these files: exit 1, MODEL_LOAD per node."""
    models = tmp_path / "models"
    models.mkdir()
    dxnn = _registry()["yolov8n"]["dxnn_file"]
    (models / dxnn).write_bytes(content)
    graph = _write_graph(tmp_path, [{"id": "od", "model": "yolov8n"}])
    result = run(BINARIES[0], "--check", str(graph), "--model-dir", str(models))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "{}  [present, invalid]".format(dxnn) in result.stdout, result.stdout
    assert 'ERROR [MODEL_LOAD] node "od": model "yolov8n" ({}) is not a loadable .dxnn: '.format(
        dxnn) in result.stderr, result.stderr
    assert "  -> download it again: {} YoloV8N --output {}\n".format(DOWNLOAD, models) \
        in result.stderr, result.stderr
    assert "setup.sh" not in result.stderr, result.stderr


SHARED_GRAPH = {
    "version": 1, "name": "shared",
    "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
              {"id": "od", "model": "yolov8-n_640x640", "engine": {"share": True, "buffers": 2}},
              {"id": "od2", "model": "yolov8-n_640x640", "engine": {"share": True, "buffers": 2}}],
    "edges": [{"from": "cam", "to": "od"}, {"from": "cam", "to": "od2"}],
}


@pytest.mark.graph_parity
def test_a_shared_engine_gives_each_node_what_it_gets_alone(tmp_path):
    """Two nodes on one engine, sync and async (8 frames in flight), give the
    same reports as two nodes with an engine each, and the same result on
    both nodes."""
    if not (MODEL_DIR / "yolov8-n_640x640.dxnn").is_file():
        pytest.skip("yolov8-n_640x640.dxnn is absent - run {} YoloV8N".format(DOWNLOAD))
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps(SHARED_GRAPH))
    unshared = json.loads(json.dumps(SHARED_GRAPH))
    for node in unshared["nodes"]:
        node.pop("engine", None)
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(unshared))
    video = write_moving_video(tmp_path / "moving.avi")
    reports = {}
    for name, graph in (("plain", plain), ("shared", shared)):
        for binary, extra in (("multi_model_graph_sync", []),
                              ("multi_model_graph_async", ["--max-inflight", "8"])):
            out = tmp_path / "{}_{}.json".format(name, binary)
            result = run(binary, "--graph", str(graph), "--input", str(video),
                         "--report", str(out), *extra)
            assert result.returncode == 0, result.stdout + result.stderr
            reports[(name, binary)] = json.loads(out.read_text())["frames"]
    first = reports[("plain", "multi_model_graph_sync")]
    assert len(first) == 8
    for key, frames in reports.items():
        assert frames == first, key
    for frame in first:
        assert frame["nodes"]["od"]["payload"] == frame["nodes"]["od2"]["payload"]


LIST_KEYS = {"model", "task", "produces", "consumes", "input", "published", "file",
             "container", "ready", "reason", "ports", "alias_of"}


def _list_json(*args):
    result = run(BINARIES[0], "--list-models", "--json", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == "", result.stderr
    return json.loads(result.stdout)


@pytest.mark.graph
def test_list_models_json_has_one_object_per_table_row():
    table = _list_rows(run(BINARIES[0], "--list-models").stdout)
    rows = _list_json()
    assert [row["model"] for row in rows] == list(table)
    assert all(set(row) == LIST_KEYS for row in rows)
    by = {row["model"]: row for row in rows}
    od = by["yolov8-n_640x640"]
    assert (od["task"], od["produces"], od["consumes"]) == ("object_detection", "boxes", "full_frame")
    assert od["input"] == {"width": 640, "height": 640}
    assert od["published"] is True and od["ready"] is True and od["reason"] == ""
    assert od["file"] == "yolov8-n_640x640.dxnn"
    assert od["container"] in table["yolov8-n_640x640"]
    assert od["ports"] == [] and od["alias_of"] is None
    assert by["yolopv2_384x640"]["ports"] == [{"name": "drivable", "shape": "labelmap"},
                                               {"name": "lane", "shape": "labelmap"}]
    alias = by["deit_base384_distilled"]
    assert alias["alias_of"] == "deit-b_384x384_distilled"
    assert alias["task"] == by["deit-b_384x384_distilled"]["task"]
    assert by["patchcore_224x224"]["ready"] is False and by["patchcore_224x224"]["reason"]


@pytest.mark.graph
def test_list_models_json_honours_the_filters():
    rows = _list_json("--consumes", "roi", "--produces", "scores")
    assert rows
    assert all(row["produces"] == "scores" and row["consumes"] in ("roi", "either") for row in rows)
    result = run(BINARIES[0], "--list-models", "--json", "--produces", "bogus_shape")
    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout == ""
    assert "--produces expects one of" in result.stderr


@pytest.mark.graph
def test_json_is_only_for_check_and_list_models():
    result = run(BINARIES[0], "--graph", "{}/cascade_od_reid_track.json".format(GRAPHS), "--json")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--json applies to --check, --list-models and --list-ops" in result.stderr


def _check_json(path, *args):
    result = run(BINARIES[0], "--check", str(path), "--json", *args)
    assert result.stderr == "", result.stderr
    return result.returncode, json.loads(result.stdout)


@pytest.mark.graph
@pytest.mark.parametrize("graph", SAMPLE_GRAPHS)
def test_check_json_of_every_sample(graph):
    code, out = _check_json(GRAPH_DIR / graph)
    assert code == 0 and out["ok"] is True and out["errors"] == [], out
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    assert out["graph"] == spec["name"]
    assert [m["node"] for m in out["models"]] == [n["id"] for n in spec["nodes"] if "model" in n]
    for model in out["models"]:
        assert model["state"] in ("present", "missing"), model
        assert model["engine"] == {"buffers": 0, "share": False, "shared_with": None}, model


CHECK_JSON_ERRORS = [
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "no_such_model_at_all"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "MODEL_UNKNOWN", "od", None),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "seg", "model": "bisenetv2"}, {"id": "reid", "model": "casvit_t"}],
      "edges": [{"from": "cam", "to": "seg"}, {"from": "seg", "to": "reid", "roi": {}}]},
     "GRAPH_EDGE", None, {"from": "seg", "to": "reid"}),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"}, {"id": "cls", "model": "resnet50"}],
      "edges": [{"from": "cam", "to": "od"}, {"from": "cam", "to": "cls"},
                {"from": "od", "to": "cls", "roi": {}}]},
     "GRAPH_EDGE", "cls", None),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"}, {"id": "od", "modle": "yolov8n"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "GRAPH_SCHEMA", "od", None),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"}, {"id": "od", "model": "yolov8n"}],
      "edges": [{"form": "cam", "to": "od"}]},
     "GRAPH_SCHEMA", None, {"from": None, "to": "od"}),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "a", "model": "yolov8n"}, {"id": "b", "model": "yolov8n"}],
      "edges": [{"from": "cam", "to": "a"}, {"from": "a", "to": "b"}, {"from": "b", "to": "a"}]},
     "GRAPH_CYCLE", None, None),
    ({"version": 1,
      "nodes": [{"id": "cam", "type": "source", "uri": "a.jpg"},
                {"id": "od", "model": "yolov8n"}, {"id": "od", "model": "resnet50"}],
      "edges": [{"from": "cam", "to": "od"}]},
     "GRAPH_SCHEMA", "od", None),
]


@pytest.mark.graph
@pytest.mark.parametrize("graph,code,node,edge", CHECK_JSON_ERRORS)
def test_check_json_pins_each_error_to_its_node_or_edge(tmp_path, graph, code, node, edge):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(graph))
    status, out = _check_json(path)
    assert status == 1 and out["ok"] is False, out
    assert len(out["errors"]) == 1, out["errors"]
    error = out["errors"][0]
    assert set(error) == {"code", "where", "what", "how", "node", "edge"}
    assert (error["code"], error["node"], error["edge"]) == (code, node, edge), error


@pytest.mark.graph
def test_check_json_of_a_file_that_is_not_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json")
    status, out = _check_json(path)
    assert status == 1 and out["ok"] is False
    assert out["errors"][0]["code"] == "GRAPH_SCHEMA" and out["errors"][0]["node"] is None
    assert "JSON" in out["errors"][0]["how"], out["errors"][0]
    assert out["models"] == []
    assert out["graph"] == str(path)


@pytest.mark.graph
def test_check_json_reports_model_files_and_engines(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    dxnn = _registry()["yolov8n"]["dxnn_file"]
    (models / dxnn).write_bytes(b"")
    graph = _write_graph(tmp_path, [
        {"id": "od", "model": "yolov8n", "engine": {"share": True, "buffers": 2}},
        {"id": "od2", "model": "yolov8-n_640x640", "engine": {"share": True, "buffers": 2}},
        {"id": "cls", "model": "resnet50"}])
    status, out = _check_json(graph, "--model-dir", str(models))
    assert status == 1, out
    assert [(e["code"], e["node"]) for e in out["errors"]] == [("MODEL_LOAD", "od"), ("MODEL_LOAD", "od2")]
    by = {m["node"]: m for m in out["models"]}
    assert by["od"]["state"] == "invalid" and by["od"]["container"] is None
    assert by["od"]["model"] == "yolov8-n_640x640" and by["od"]["file"] == dxnn
    assert by["od"]["engine"] == {"buffers": 2, "share": True, "shared_with": None}
    assert by["od2"]["engine"] == {"buffers": 2, "share": True, "shared_with": "od"}
    assert by["cls"]["state"] == "missing"
    assert by["cls"]["engine"] == {"buffers": 0, "share": False, "shared_with": None}
    assert any('"yolov8n" is the old name of "yolov8-n_640x640"' in note for note in out["notes"])


@pytest.mark.graph
@pytest.mark.parametrize("args", [
    ["--check", "--json"],
    ["--json", "--check"],
    ["--list-models", "--json", "--check", "x.json"],
    ["--check", "x.json", "--json", "--frames", "abc"],
], ids=["check-json-no-path", "json-check-no-path", "list-models-and-check", "bad-value"])
def test_json_usage_errors_keep_stdout_empty(args):
    """--json promises one JSON document on stdout: a usage error prints its
    message and the usage text on stderr, exit 2, and nothing on stdout."""
    for binary in BINARIES:
        result = run(binary, *args)
        assert result.returncode == 2, result.stdout + result.stderr
        assert result.stdout == "", result.stdout
        assert "Usage:" in result.stderr, result.stderr


@pytest.mark.graph
def test_list_models_and_check_are_separate_modes():
    result = run(BINARIES[0], "--list-models", "--check", "{}/{}".format(GRAPHS, MULTISTREAM))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--list-models and --check" in result.stderr, result.stderr


@pytest.mark.graph
def test_check_json_duplicate_node_id_names_the_node_index(tmp_path):
    graph = _write_graph(tmp_path, [{"id": "od", "model": "yolov8n"},
                                    {"id": "od", "model": "resnet50"}],
                         edges=[{"from": "cam", "to": "od"}])
    status, out = _check_json(graph)
    assert status == 1
    error = out["errors"][0]
    assert error["where"].endswith(' node[2] "od"'), error
    assert error["node"] == "od", error


@pytest.mark.graph
@pytest.mark.parametrize("reserved", ["base", "events"])
def test_check_rejects_the_node_ids_layer_files_are_named_after(tmp_path, reserved):
    """--output-layers writes <frame>_<node id>.png beside <frame>_base.png and
    <frame>_events.png: a node called "base" or "events" would overwrite one of
    them, so --check refuses the id, pinned to the node."""
    graph = _write_graph(tmp_path, [{"id": reserved, "model": "yolov8n"}])
    status, out = _check_json(graph)
    assert status == 1, out
    error = out["errors"][0]
    assert error["code"] == "GRAPH_SCHEMA", error
    assert error["node"] == reserved, error
    assert 'node id "{}" is reserved'.format(reserved) in error["what"], error
    assert error["how"], error
    result = run(BINARIES[0], "--check", str(graph))
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'node id "{}" is reserved'.format(reserved) in result.stdout + result.stderr


# ---------------------------------------------------------------- graph composer P2

# A tracked person detector, a person filter, a right-half zone and an alarm.
# On sample_people.jpg the zone holds one of the two people.
INTRUSION_GRAPH = {
    "version": 1, "name": "intrusion",
    "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
              {"id": "od", "model": "yolov8-n_640x640", "params": {"score_threshold": 0.35},
               "track": {"algo": "iou", "iou": 0.3, "max_age": 30}},
              {"id": "person", "type": "logic", "op": "filter",
               "params": {"classes": ["person"]}},
              {"id": "zone", "type": "logic", "op": "zone",
               "params": {"polygon": [0.45, 0.35, 1.0, 0.35, 1.0, 1.0, 0.45, 1.0]}},
              {"id": "alarm", "type": "logic", "op": "condition",
               "params": {"value": "count", "comparator": ">", "threshold": 0,
                          "event": "intrusion"}}],
    "edges": [{"from": "cam", "to": "od"},
              {"from": "od", "to": "person", "carry": "result"},
              {"from": "person", "to": "zone", "carry": "result"},
              {"from": "zone", "to": "alarm", "carry": "result"}],
}


@pytest.mark.graph
def test_check_lists_logic_nodes(tmp_path):
    path = tmp_path / "intrusion.json"
    path.write_text(json.dumps(INTRUSION_GRAPH))
    result = run(BINARIES[0], "--check", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OK: graph "intrusion" is valid (5 nodes, 4 edges, 1 models)' in result.stdout
    rows = {line.split()[0]: line for line in result.stdout.splitlines() if line.split()}
    assert "(logic zone)" in rows["zone"] and " boxes " in rows["zone"], rows["zone"]
    assert "result" in rows["zone"], rows["zone"]
    assert "(logic condition)" in rows["alarm"] and " records " in rows["alarm"], rows["alarm"]
    code, out = _check_json(path)
    assert code == 0 and out["ok"] is True, out
    assert [m["node"] for m in out["models"]] == ["od"]


@pytest.mark.graph
def test_check_json_pins_a_logic_error_to_its_node(tmp_path):
    graph = json.loads(json.dumps(INTRUSION_GRAPH))
    graph["nodes"][3]["params"] = {"polygon": [0.1, 0.1, 0.9, 0.1]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(graph))
    code, out = _check_json(path)
    assert code == 1, out
    error = out["errors"][0]
    assert error["code"] == "GRAPH_SCHEMA" and error["node"] == "zone", error
    assert '"polygon" must hold at least 3 points' in error["what"], error


def _filtered_headpose(min_score):
    return {
        "version": 1, "name": "filtered_headpose",
        "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_face.jpg"},
                  {"id": "face", "model": "scrfd-500m_640x640"},
                  {"id": "f", "type": "logic", "op": "filter",
                   "params": {"min_score": min_score}},
                  {"id": "headpose", "type": "cpu", "op": "headpose"}],
        "edges": [{"from": "cam", "to": "face"},
                  {"from": "face", "to": "f", "carry": "result"},
                  {"from": "f", "to": "headpose", "carry": "result"}],
    }


@pytest.mark.graph_parity
def test_a_cpu_node_behind_a_filter_gets_only_the_filtered_boxes(tmp_path):
    """A logic node's boxes reach a cpu node in both executors: with every
    face kept, headpose equals the plain dms_headpose graph's; with none kept,
    it writes no row."""
    absent = missing_artifacts("dms_headpose.json")
    if absent:
        pytest.skip(absent)
    plain = tmp_path / "plain.json"
    result = run("multi_model_graph_sync", "--graph", "{}/dms_headpose.json".format(GRAPHS),
                 "--report", str(plain))
    assert result.returncode == 0, result.stdout + result.stderr
    direct = json.loads(plain.read_text())["frames"][0]["nodes"]["headpose"]
    rows = {}
    for min_score in (0.5, 0.99):
        graph = tmp_path / "filtered_{}.json".format(min_score)
        graph.write_text(json.dumps(_filtered_headpose(min_score)))
        reports = {}
        for binary in ("multi_model_graph_sync", "multi_model_graph_async"):
            out = tmp_path / "{}_{}.json".format(binary, min_score)
            result = run(binary, "--graph", str(graph), "--report", str(out))
            assert result.returncode == 0, result.stdout + result.stderr
            reports[binary] = json.loads(out.read_text())["frames"]
        assert reports["multi_model_graph_sync"] == reports["multi_model_graph_async"]
        frame = reports["multi_model_graph_sync"][0]
        assert frame["error"] == "", frame["error"]
        rows[min_score] = frame["nodes"]["headpose"]
        kept = frame["nodes"]["f"]["payload"]["items"]
        assert all(item["score"] >= min_score for item in kept)
    assert rows[0.5]["payload"] == direct["payload"]
    assert rows[0.99]["payload"]["items"] == []


def _needs_model(name):
    """Skip unless assets/models/<name>.dxnn is a real file (the host keeps
    0-byte placeholders for models it has not downloaded)."""
    path = MODEL_DIR / (name + ".dxnn")
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("{} is absent - run {} {}".format(path, DOWNLOAD, name))


def _event_lines(stdout):
    return [json.loads(line[len("[EVENT] "):]) for line in stdout.splitlines()
            if line.startswith("[EVENT] ")]


@pytest.mark.graph
@pytest.mark.parametrize("mode", [("--check",), ("--list-models",)])
def test_events_apply_to_a_run(tmp_path, mode):
    path = tmp_path / "intrusion.json"
    path.write_text(json.dumps(INTRUSION_GRAPH))
    args = list(mode) + ([str(path)] if mode[0] == "--check" else [])
    result = run(BINARIES[0], *(args + ["--events", str(tmp_path / "e.jsonl")]))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--events applies to a run (--graph), not to a listing or --check" in result.stderr


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_events_go_to_stdout_the_events_file_and_the_report(tmp_path, binary):
    _needs_model("yolov8-n_640x640")
    graph = tmp_path / "intrusion.json"
    graph.write_text(json.dumps(INTRUSION_GRAPH))
    events, report = tmp_path / "events.jsonl", tmp_path / "report.json"
    result = run(binary, "--graph", str(graph), "--events", str(events), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    printed = _event_lines(result.stdout)
    assert [(e["node"], e["event"]) for e in printed] == [("zone", "enter"), ("alarm", "intrusion")]
    assert list(printed[0]) == ["frame", "time", "stream", "node", "event", "track_id", "data"]
    assert printed[0]["frame"] == 0 and printed[0]["time"] == 0 and printed[0]["stream"] == "cam"
    assert printed[0]["track_id"] >= 0 and printed[0]["data"] == {"count": 1}
    assert printed[1]["track_id"] == printed[0]["track_id"]
    assert printed[1]["data"] == {"value": 1, "threshold": 0,
                                  "track_ids": str(printed[0]["track_id"])}
    assert [json.loads(line) for line in events.read_text().splitlines()] == printed
    frame = json.loads(report.read_text())["frames"][0]
    assert [(e["node"], e["event"]) for e in frame["events"]] == [("zone", "enter"),
                                                                  ("alarm", "intrusion")]
    assert len(frame["nodes"]["zone"]["payload"]["items"]) == 1
    lines = result.stdout.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("frame 0:"))
    assert lines[first + 1].startswith("[EVENT] "), lines[first:first + 3]
    assert "zone boxes=1" in lines[first], lines[first]


@pytest.mark.graph_parity
def test_sync_and_async_write_the_same_events(tmp_path):
    _needs_model("yolov8-n_640x640")
    graph = tmp_path / "intrusion.json"
    graph.write_text(json.dumps(INTRUSION_GRAPH))
    video = write_moving_video(tmp_path / "moving.avi")
    files = {}
    for binary in BINARIES:
        files[binary] = tmp_path / (binary + ".jsonl")
        result = run(binary, "--graph", str(graph), "--input", str(video),
                     "--events", str(files[binary]))
        assert result.returncode == 0, result.stdout + result.stderr
    assert files[BINARIES[0]].read_bytes() == files[BINARIES[1]].read_bytes()
    assert files[BINARIES[0]].read_text().count("\n") >= 2


@pytest.mark.graph
def test_loop_with_per_frame_images_needs_frames(tmp_path):
    result = run(BINARIES[0], "--graph", "{}/cascade_od_reid_track.json".format(GRAPHS),
                 "--input", str(tmp_path / "walk.mp4"), "--loop",
                 "--output", str(tmp_path / "out.png"))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--loop never ends a video" in result.stderr, result.stderr


@pytest.mark.graph_e2e
def test_loop_on_an_image_writes_its_one_image(tmp_path):
    """--loop restarts only videos: an image source ends after its frame, so
    per-frame image --output without --frames is fine."""
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    out = tmp_path / "out.png"
    result = run(BINARIES[0], "--graph", "{}/{}".format(GRAPHS, graph), "--loop",
                 "--output", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.is_file() and "1 frame, 0 failed" in result.stdout.splitlines(), result.stdout


@pytest.mark.graph
def test_loop_applies_to_a_run():
    result = run(BINARIES[0], "--check", "{}/cascade_od_reid_track.json".format(GRAPHS), "--loop")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--loop applies to a run (--graph), not to a listing or --check" in result.stderr


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_loop_restarts_a_video_until_frames(tmp_path, binary):
    graph = "cascade_od_reid_track.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    video = write_moving_video(tmp_path / "moving.avi", count=4)
    report = tmp_path / "report.json"
    result = run_env(binary, {"DXAPP_PROGRESS": "1"}, "--graph", "{}/{}".format(GRAPHS, graph),
                     "--input", str(video), "--loop", "--frames", "10", "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    frames = json.loads(report.read_text())["frames"]
    assert [frame["index"] for frame in frames] == list(range(10))
    boxes = [[item["box"] for item in frame["nodes"]["od"]["payload"]["items"]] for frame in frames]
    assert boxes[4] == boxes[0] and boxes[8] == boxes[0], "frame 4 and 8 are frame 0 again"
    # Each restart starts the tracker again: the same ids as on the first pass.
    tracks = [[item["track_id"] for item in frame["nodes"]["od"]["payload"]["items"]]
              for frame in frames]
    assert tracks[4] == tracks[0] and tracks[8] == tracks[0], tracks
    assert "10 frames, 0 failed" in result.stdout.splitlines(), result.stdout[-300:]
    assert _field(_progress_lines(result.stdout)[-1], "frames") == "10"


OP_KEYS = {"op", "type", "summary", "inputs", "output", "tracking", "events", "numbers", "params"}


@pytest.mark.graph
def test_list_ops_json_describes_every_op():
    result = run(BINARIES[0], "--list-ops", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    ops = json.loads(result.stdout)
    assert [op["op"] for op in ops] == ["filter", "zone", "line", "count", "dwell",
                                        "condition", "gallery", "user", "headpose", "volume"]
    assert all(set(op) == OP_KEYS for op in ops)
    # In the order the README documents, so a person can read it.
    assert list(ops[1]) == ["op", "type", "summary", "inputs", "output", "tracking", "events",
                            "numbers", "params"]
    assert list(ops[1]["params"][0]) == ["name", "type", "required", "min_points", "help"]
    by = {op["op"]: op for op in ops}
    assert by["zone"]["type"] == "logic" and by["volume"]["type"] == "cpu"
    assert by["line"]["tracking"] is True and by["zone"]["tracking"] is False
    assert by["count"]["inputs"]["max"] == 8 and by["volume"]["inputs"]["min"] == 3
    assert {"shape": "scores", "per_crop": True} in by["condition"]["inputs"]["accepts"]
    assert by["filter"]["output"] == "same" and by["line"]["output"] == "records"
    params = {p["name"]: p for p in by["zone"]["params"]}
    assert params["polygon"] == {"name": "polygon", "type": "polygon", "required": True,
                                 "min_points": 3, "help": params["polygon"]["help"]}
    assert params["mode"]["default"] == "inside" and params["mode"]["choices"] == ["inside",
                                                                                   "outside"]
    dwell = {p["name"]: p for p in by["dwell"]["params"]}
    assert dwell["threshold_s"]["min"] == 0 and dwell["threshold_s"]["min_exclusive"] is True
    assert "max" not in dwell["threshold_s"] and dwell["threshold_s"]["default"] == 10
    condition = {p["name"]: p for p in by["condition"]["params"]}
    assert "min" not in condition["threshold"] and "max" not in condition["threshold"]
    assert condition["value"]["default"] == "count"
    assert by["condition"]["events"] == ["<event>"]
    assert by["line"]["numbers"] == ["in", "out", "total"]


@pytest.mark.graph
def test_list_ops_json_gives_an_inspector_its_hints():
    """Fields a GUI inspector needs beyond type and bounds; each appears only
    where it applies, so every other param keeps its keys."""
    ops = {op["op"]: op for op in json.loads(run(BINARIES[0], "--list-ops", "--json").stdout)}

    def params(op):
        return {p["name"]: p for p in ops[op]["params"]}

    # A list left out is not [] (volume: the built-in package classes; [] matches none).
    volume = params("volume")
    for name in ("classes", "class_ids"):
        assert "default" not in volume[name], volume[name]
    assert volume["classes"]["items"] == "string" and volume["class_ids"]["items"] == "integer"
    assert volume["classes"]["when_omitted"] == ["backpack", "handbag", "suitcase", "box",
                                                 "package", "luggage"]
    assert volume["class_ids"]["when_omitted"] == ["24", "26", "28"]
    assert "default" not in params("filter")["classes"]
    assert "when_omitted" not in params("filter")["classes"]
    gallery = params("gallery")["gallery"]
    assert gallery["type"] == "text" and gallery["format"] == "file"
    assert gallery["extensions"] == [".bin"]
    assert params("filter")["min_score"]["step"] == 0.01
    assert params("dwell")["threshold_s"]["step"] == 0.1
    assert params("user")["output"]["equals"] == "input"
    assert "equals" not in params("user")["input"]
    by_input = params("condition")["value"]["values_by_input"]
    assert {"shape": "boxes", "per_crop": False, "values": ["count"]} in by_input
    assert {"shape": "records", "per_crop": False, "values_from": "numbers"} in by_input
    assert {"shape": "scores", "per_crop": True,
            "values": ["count", "with:<label>", "without:<label>"]} in by_input
    assert {"shape": "vector", "per_crop": True, "values": ["count"]} in by_input
    help_last = [p for op in ops.values() for p in op["params"] if list(p)[-1] != "help"]
    assert help_last == []


@pytest.mark.graph
def test_list_ops_prints_a_table():
    result = run(BINARIES[0], "--list-ops")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].split()[:5] == ["op", "type", "inputs", "output", "tracking"]
    rows = {line.split()[0]: line.split() for line in lines[1:] if line.strip()}
    assert rows["count"][:5] == ["count", "logic", "1-8", "records", "no"]
    assert rows["line"][:5] == ["line", "logic", "1", "records", "yes"]
    assert rows["volume"][:4] == ["volume", "cpu", "3", "records"]


@pytest.mark.graph
def test_list_ops_is_a_mode_of_its_own():
    result = run(BINARIES[0], "--list-ops", "--list-models")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--list-ops is a mode of its own" in result.stderr


# What each logic sample decides on its own sample image (sample_people.jpg:
# two people, a dog; the last three read their own images, see their own
# tests). Node ids and event names as in the sample files.
LOGIC_SAMPLE_EVENTS = {
    "zone_intrusion.json": [("zone", "enter"), ("alarm", "intrusion")],
    "line_counting.json": [],
    "ppe_zone_check.json": [("zone", "enter"), ("alarm", "no_hat")],
    "loitering_dwell.json": [("zone", "enter")],
    "zone_people_count.json": [("left", "enter"), ("right", "enter")],
    "gallery_known_person.json": [("known", "match"), ("known", "match")],
    "driver_attention.json": [],
    "multistream_people_count.json": [("crowded", "crowded")],
    "logistics_volume_alert.json": [("load", "load_high")],
}


@pytest.mark.graph_e2e
@pytest.mark.parametrize("graph", sorted(LOGIC_SAMPLE_EVENTS))
def test_logic_samples_decide_on_their_sample_image(tmp_path, graph):
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report = tmp_path / "report.json"
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
                 "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    events = _event_lines(result.stdout)
    assert [(e["node"], e["event"]) for e in events] == LOGIC_SAMPLE_EVENTS[graph], result.stdout
    frame = json.loads(report.read_text())["frames"][0]
    if graph == "line_counting.json":
        numbers = frame["nodes"]["line"]["payload"]["items"][0]["numbers"]
        assert numbers == {"in": 0, "out": 0, "total": 0}, numbers
    if graph == "zone_people_count.json":
        items = frame["nodes"]["count"]["payload"]["items"]
        assert {i["text"]["node"]: i["numbers"]["total"] for i in items} == {"left": 1, "right": 1}
    if graph == "ppe_zone_check.json":
        assert len(frame["roi_nodes"]["attr"]) == 1, "one person stands in the zone"
    if graph == "gallery_known_person.json":
        assert sorted(e["data"]["name"] for e in events) == ["sample_people_p0", "sample_people_p1"]
        assert all(e["data"]["score"] > 0.9 for e in events), events


@pytest.mark.graph
def test_readme_documents_logic_nodes_and_events():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Logic nodes", 1)[1].split("\n---", 1)[0]
    ops = json.loads(run(BINARIES[0], "--list-ops", "--json").stdout)
    for op in ops:
        if op["type"] == "logic":
            assert "| `{}` |".format(op["op"]) in section, op["op"]
    for needle in ("[EVENT] ", "--events", "--list-ops --json", "cooldown_s", "0 to 1",
                   "per crop", "with:<label>", "build_gallery_database.py", "track_id",
                   "seconds since"):
        assert needle in section, needle
    assert "## The whole vocabulary: four node kinds, three edge kinds" in text
    assert '| logic | `{ "id": "zone", "type": "logic"' in text
    py = DX_GRAPH_README.read_text(encoding="utf-8")
    assert '"events"' in py and "--loop" in py


# ---------------------------------------------------------------------
# Results for front ends (spec 16): --draw, --output-layers, the grid.
# ---------------------------------------------------------------------

FANOUT = "fanout_od_seg_depth.json"


@pytest.mark.graph
@pytest.mark.parametrize("binary", BINARIES)
def test_help_names_the_results_options(binary):
    result = run(binary, "--help")
    assert result.returncode == 0
    assert "--draw <id[,id...]>" in result.stdout
    assert "--output-layers <dir>" in result.stdout
    assert "layers.json" in result.stdout


@pytest.mark.graph
@pytest.mark.parametrize("value, needle", [
    ("od,nope", '--draw: "nope" is not a node of this graph that draws; '
                "its nodes: od, seg, depth"),
    ("cam", '--draw: "cam" is not a node of this graph that draws; '
            "its nodes: od, seg, depth"),
    (",", "--draw needs node ids, separated by commas"),
])
def test_draw_names_nodes_of_the_graph(value, needle):
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, FANOUT),
                 "--draw", value)
    assert result.returncode == 2, result.stdout + result.stderr
    assert needle in result.stderr
    assert "graph:" not in result.stdout  # nothing was built


@pytest.mark.graph
@pytest.mark.parametrize("option", [("--draw", "od"), ("--output-layers", "layers")])
def test_results_options_apply_to_a_run(tmp_path, option):
    result = run(BINARIES[0], "--check", "{}/{}".format(GRAPHS, FANOUT), option[0],
                 str(tmp_path / option[1]) if option[0] == "--output-layers" else option[1])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "{} applies to a run (--graph), not to a listing or --check".format(option[0]) \
        in result.stderr


@pytest.mark.graph
@pytest.mark.parametrize("uri", ["video", "camera:0", "rtsp://127.0.0.1:1/none"])
def test_layers_of_a_video_or_a_live_source_need_frames(tmp_path, uri):
    if uri == "video":
        uri = str(tmp_path / "clip.avi")  # only its extension is read before the refusal
    layers = tmp_path / "layers"
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, FANOUT),
                 "--input", uri, "--output-layers", str(layers))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--output-layers writes" in result.stderr
    assert "stop after N frames (--frames N)" in result.stderr
    assert "graph:" not in result.stdout
    assert not layers.exists()


@pytest.mark.graph
def test_layers_into_a_file_fails_before_models_load(tmp_path):
    taken = tmp_path / "file"
    taken.write_text("x")
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, FANOUT),
                 "--output-layers", str(taken))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not write {}: it is not a directory".format(taken) in result.stderr
    assert "graph:" not in result.stdout


def _read_rgba(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, path
    return image


def _stack(base, layers):
    """What a front end does: every RGBA layer over the base, in order."""
    import numpy as np
    out = base.astype(np.float64)
    for layer in layers:
        alpha = layer[:, :, 3:4].astype(np.float64) / 255.0
        out = layer[:, :, :3] * alpha + out * (1.0 - alpha)
    return np.rint(out).astype(np.uint8)


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", BINARIES)
def test_output_layers_stack_back_to_the_output(tmp_path, binary):
    import cv2
    import numpy as np
    absent = missing_artifacts(FANOUT)
    if absent:
        pytest.skip(absent)
    output, layers = tmp_path / "result.png", tmp_path / "layers"
    result = run(binary, "--graph", "{}/{}".format(GRAPHS, FANOUT), "--output", str(output),
                 "--output-layers", str(layers))
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((layers / "layers.json").read_text())
    assert list(manifest) == ["frames"] and len(manifest["frames"]) == 1
    frame = manifest["frames"][0]
    assert list(frame) == ["stream", "frame", "base", "layers"]
    assert (frame["stream"], frame["frame"]) == ("cam", 0)
    assert frame["base"] == "result_000000_base.png"
    # RenderReport's pass order: seg's labelmap, depth's densemap, od's boxes.
    assert frame["layers"] == [
        {"node": "seg", "model": "bisenetv2_1024x2048", "file": "result_000000_seg.png"},
        {"node": "depth", "model": "fastdepth_224x224", "file": "result_000000_depth.png"},
        {"node": "od", "model": "yolov8-n_640x640", "file": "result_000000_od.png"},
    ]
    assert sorted(p.name for p in layers.iterdir()) == sorted(
        ["layers.json", frame["base"]] + [entry["file"] for entry in frame["layers"]])
    assert "layers: {}".format(layers / "layers.json") in result.stdout
    base = cv2.imread(str(layers / frame["base"]))
    source = cv2.imread(str(PROJECT_ROOT / "sample" / "img" / "sample_street.jpg"))
    assert base.shape == source.shape and np.array_equal(base, source)
    rgba = [_read_rgba(layers / entry["file"]) for entry in frame["layers"]]
    assert all(layer.shape == source.shape[:2] + (4,) for layer in rgba)
    od_alpha = rgba[2][:, :, 3]
    assert 0 < np.count_nonzero(od_alpha) < od_alpha.size // 2  # boxes, not the whole frame
    merged = cv2.imread(str(output))
    diff = np.abs(_stack(base, rgba).astype(int) - merged.astype(int))
    assert diff.max() <= 3, diff.max()


@pytest.mark.graph_e2e
def test_draw_draws_only_the_named_nodes(tmp_path):
    import cv2
    import numpy as np
    absent = missing_artifacts(FANOUT)
    if absent:
        pytest.skip(absent)
    everything, layers = tmp_path / "all.png", tmp_path / "layers"
    a = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, FANOUT),
            "--output", str(everything), "--output-layers", str(layers))
    od, od_layers, report = tmp_path / "od.png", tmp_path / "od_layers", tmp_path / "r.json"
    b = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, FANOUT),
            "--draw", "od", "--output", str(od), "--output-layers", str(od_layers),
            "--report", str(report))
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    # Every node still ran and reported.
    assert sorted(json.loads(report.read_text())["frames"][0]["nodes"]) == ["cam", "depth", "od", "seg"]
    frame = json.loads((od_layers / "layers.json").read_text())["frames"][0]
    assert [entry["node"] for entry in frame["layers"]] == ["od"]
    # The image is the clean frame with od's layer alone over it.
    base = cv2.imread(str(layers / "all_000000_base.png"))
    od_layer = _read_rgba(layers / "all_000000_od.png")
    diff = np.abs(_stack(base, [od_layer]).astype(int) - cv2.imread(str(od)).astype(int))
    assert diff.max() <= 3, diff.max()


@pytest.mark.graph_e2e
def test_layers_of_an_event_frame_end_with_the_event_panel(tmp_path):
    _needs_model("yolov8-n_640x640")
    graph = tmp_path / "intrusion.json"
    graph.write_text(json.dumps(INTRUSION_GRAPH))
    output, layers = tmp_path / "out.png", tmp_path / "layers"
    result = run("multi_model_graph_async", "--graph", str(graph), "--output", str(output),
                 "--output-layers", str(layers))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.loads((layers / "layers.json").read_text())["frames"][0]
    assert frame["layers"][0] == {"node": "od", "model": "yolov8-n_640x640",
                                  "file": "out_000000_od.png"}
    assert {"node": "zone", "op": "zone", "file": "out_000000_zone.png"} in frame["layers"]
    assert frame["layers"][-1] == {"panel": "events", "file": "out_000000_events.png"}
    import cv2
    import numpy as np
    base = cv2.imread(str(layers / frame["base"]))
    rgba = [_read_rgba(layers / entry["file"]) for entry in frame["layers"]]
    diff = np.abs(_stack(base, rgba).astype(int) - cv2.imread(str(output)).astype(int))
    assert diff.max() <= 3, diff.max()
    # --draw od: the zone's events are not od's, so no panel either.
    drawn = tmp_path / "drawn"
    result = run("multi_model_graph_async", "--graph", str(graph), "--draw", "od",
                 "--output-layers", str(drawn))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.loads((drawn / "layers.json").read_text())["frames"][0]
    assert frame["base"] == "intrusion_000000_base.png"  # no --output: the graph's stem
    assert [entry.get("node", entry.get("panel")) for entry in frame["layers"]] == ["od"]


@pytest.mark.graph_e2e
@pytest.mark.parametrize("binary", ["multi_model_graph_sync", "multi_model_graph_async"])
def test_driver_attention_draws_the_head_pose_on_the_face(tmp_path, binary):
    """The headpose node has a layer of its own: the head's axes from the nose, inside the face box, and its
    angles beside it. Before, the overlay was the face box alone."""
    graph = "driver_attention.json"
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)
    report, layers = tmp_path / "report.json", tmp_path / "layers"
    result = run(binary, "--graph", "{}/{}".format(GRAPHS, graph), "--report", str(report),
                 "--output-layers", str(layers))
    assert result.returncode == 0, result.stdout + result.stderr
    frame = json.loads((layers / "layers.json").read_text())["frames"][0]
    entry = [e for e in frame["layers"] if e.get("node") == "headpose"]
    assert entry == [{"node": "headpose", "op": "headpose", "file": entry[0]["file"] if entry else ""}]
    x, y, w, h = json.loads(report.read_text())["frames"][0]["nodes"]["face"]["payload"]["items"][0]["box"]
    rgba = _read_rgba(layers / entry[0]["file"])
    face = rgba[int(y):int(y + h), int(x):int(x + w)]
    drawn = face[..., 3] > 0
    red = drawn & (face[..., 2] > 200) & (face[..., 1] < 80) & (face[..., 0] < 80)
    green = drawn & (face[..., 1] > 150) & (face[..., 2] < 80) & (face[..., 0] < 80)
    assert red.sum() > 200 and green.sum() > 200, (red.sum(), green.sum())   # thick axes, not a dot


@pytest.mark.graph_e2e
def test_layers_of_several_streams_name_their_source(tmp_path):
    absent = missing_artifacts(MULTISTREAM)
    if absent:
        pytest.skip(absent)
    cam1, cam2 = write_two_stream_videos(tmp_path)
    layers = tmp_path / "layers"
    result = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2),
                 "--frames", "2", "--output-layers", str(layers))
    assert result.returncode == 0, result.stdout + result.stderr
    frames = json.loads((layers / "layers.json").read_text())["frames"]
    assert sorted((f["stream"], f["frame"]) for f in frames) == [
        ("cam1", 0), ("cam1", 1), ("cam2", 0), ("cam2", 1)]
    for frame in frames:
        assert frame["base"] == "multistream_od_two_sources_{}_{:06d}_base.png".format(
            frame["stream"], frame["frame"])
        assert (layers / frame["base"]).is_file()
        for entry in frame["layers"]:
            assert (layers / entry["file"]).is_file()
