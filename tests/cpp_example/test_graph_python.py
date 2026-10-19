"""dx_graph: the Python binding of the multi-model graph engine."""
import importlib
import json
import os
import platform
import sys

import pytest

from conftest import PROJECT_ROOT, resolve_bin_dir
from test_graph_cli import (ALIAS_NOTES, GRAPHS, GRAPH_DIR, MODEL_DIR, MULTISTREAM,
                            RUNNABLE_SAMPLE_GRAPHS, SAMPLE_GRAPHS,
                            TWO_SR_GRAPH, V9_MODELS, _registry, missing_artifacts, one_node_graph, read_in_turn,
                            require_an_older_runtime, run, shifted_sample_frames, v9_model_dir,
                            write_moving_video, write_two_stream_videos)

PY_DIR = os.environ.get("DX_GRAPH_PYTHONPATH") or str(resolve_bin_dir() / "python")


def load_dx_graph():
    """dx_graph from PY_DIR. Skipped only when it is absent (not built, not
    shipped); a package that is there and does not import - built for
    another interpreter, a broken copy - FAILS with its ImportError."""
    if not os.path.isdir(os.path.join(PY_DIR, "dx_graph")):
        pytest.skip("dx_graph not built: no {}; build it with "
                    "./build.sh --all --python_exec <python>".format(
                        os.path.join(PY_DIR, "dx_graph")))
    if PY_DIR not in sys.path:
        sys.path.insert(0, PY_DIR)
    try:
        return importlib.import_module("dx_graph")
    except ImportError as error:
        pytest.fail("{} is there but does not import: {}".format(
            os.path.join(PY_DIR, "dx_graph"), error), pytrace=False)


@pytest.mark.graph
def test_import_reports_its_build():
    dx_graph = load_dx_graph()
    info = dx_graph.build_info()
    # Built for the interpreter that runs the tests (major.minor).
    assert info["python"].startswith("{}.{}.".format(*sys.version_info[:2]))
    assert os.path.realpath(info["project_root"]) == os.path.realpath(str(PROJECT_ROOT))
    assert isinstance(dx_graph.__version__, str) and dx_graph.__version__


def import_in_subprocess(package_dir):
    """`import dx_graph` from `package_dir` in a fresh interpreter (this
    one's executable) -> CompletedProcess."""
    import subprocess
    env = dict(os.environ, PYTHONPATH=str(package_dir))
    return subprocess.run([sys.executable, "-c", "import dx_graph"], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=60)


@pytest.mark.graph
def test_import_error_names_the_interpreter_and_the_files(tmp_path):
    """Minor #3: a _dx_graph that cannot be loaded is reported as that -
    this interpreter's sys.version and the _dx_graph*.so files present - and
    not as Python's "partially initialized module ... circular import"."""
    import importlib.machinery
    load_dx_graph()  # the shipped package, whose __init__.py is copied
    source = os.path.join(PY_DIR, "dx_graph", "__init__.py")
    ours = "_dx_graph" + importlib.machinery.EXTENSION_SUFFIXES[0]
    cases = (
        ([], "The package has no compiled core"),
        (["_dx_graph.cpython-39-x86_64-linux-gnu.so"], "was built for another interpreter"),
        ([ours], "Loading {} failed: ".format(ours)),  # not an ELF file at all
    )
    for k, (files, says) in enumerate(cases):
        package = tmp_path / str(k) / "dx_graph"
        package.mkdir(parents=True)
        with open(source, encoding="utf-8") as handle:
            (package / "__init__.py").write_text(handle.read(), encoding="utf-8")
        for name in files:
            (package / name).write_bytes(b"not a shared object")
        result = import_in_subprocess(package.parent)
        assert result.returncode != 0
        last = result.stderr.strip().splitlines()[-1]
        assert last.startswith("ImportError: dx_graph: cannot load its C++ core"), result.stderr
        assert sys.version in last, last
        for name in files:
            assert name in last, last
        assert says in last, last
        assert "circular import" not in result.stderr, result.stderr


# ------------------------------------------------------------- Graph / run
#
# Tiers as in test_graph_cli.py: ``graph`` needs no NPU (every case fails
# before a model is opened); ``graph_e2e`` builds a real graph and skips for
# exactly one reason, a .dxnn the graph needs is not downloaded.

FRAME_RULE = "frame must be a numpy uint8 array of shape (H, W, 3), got "


def require_artifacts(graph):
    absent = missing_artifacts(graph)
    if absent:
        pytest.skip(absent)


def graph_source_image(graph):
    """The graph's own source image, decoded exactly as the CLI decodes it
    (cv::imread, IMREAD_COLOR - the same system OpenCV)."""
    import cv2
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    uri = next(node["uri"] for node in spec["nodes"] if node.get("type") == "source")
    image = cv2.imread(str(PROJECT_ROOT / uri), cv2.IMREAD_COLOR)
    assert image is not None, uri
    return image


def graph_sources(graph):
    """(source id, image decoded as the CLI decodes it) for every source node,
    in declaration order."""
    import cv2
    with open(str(GRAPH_DIR / graph), encoding="utf-8") as handle:
        spec = json.load(handle)
    out = []
    for node in spec["nodes"]:
        if node.get("type") == "source":
            image = cv2.imread(str(PROJECT_ROOT / node["uri"]), cv2.IMREAD_COLOR)
            assert image is not None, node["uri"]
            out.append((node["id"], image))
    return out


@pytest.fixture
def cli_single_image_report(tmp_path):
    """Run a CLI binary on a graph's own (single-image) source with --report
    and return the report's ``frames`` list."""
    def produce(graph, binary):
        report = tmp_path / "{}_{}".format(binary, graph)
        result = run(binary, "--graph", "{}/{}".format(GRAPHS, graph),
                     "--report", str(report))
        assert result.returncode == 0, result.stdout + result.stderr
        with open(str(report), encoding="utf-8") as handle:
            return json.load(handle)["frames"]
    return produce


def find_arrays(node, found):
    """Every (summary, array) pair to_dict() attached, anywhere in the dict."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_array"):
                found.append((node[key[:-len("_array")]], value))
            else:
                find_arrays(value, found)
    elif isinstance(node, list):
        for value in node:
            find_arrays(value, found)
    return found


@pytest.mark.graph
def test_graph_error_matches_cli(tmp_path):
    dx_graph = load_dx_graph()
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"version":1,"name":"b","nodes":['
        '{"id":"cam","type":"source","uri":"sample/img/sample_people.jpg"},'
        '{"id":"seg","model":"bisenetv2"},{"id":"od","model":"yolov5n"}],'
        '"edges":[{"from":"cam","to":"seg"},{"from":"seg","to":"od"}]}')
    cli = run("multi_model_graph_sync", "--graph", str(bad), "--check")
    assert cli.returncode == 1
    with pytest.raises(dx_graph.GraphError) as caught:
        dx_graph.Graph(str(bad))
    assert caught.value.code == "GRAPH_EDGE"
    assert str(caught.value) == cli.stderr.strip()
    assert isinstance(caught.value, ValueError)


@pytest.mark.graph
@pytest.mark.parametrize("model", V9_MODELS)
def test_v9_model_on_an_older_runtime_raises_before_loading(tmp_path, model):
    """R12 through dx_graph: the CLI's MODEL_LOAD message, word for word,
    raised before any engine is created."""
    require_an_older_runtime()
    dx_graph = load_dx_graph()
    model_dir, dxnn = v9_model_dir(tmp_path, model)
    graph = one_node_graph(tmp_path, model)
    cli = run("multi_model_graph_sync", "--graph", str(graph), "--model-dir", str(model_dir))
    assert cli.returncode == 1
    with pytest.raises(dx_graph.GraphError) as caught:
        dx_graph.Graph(str(graph), model_dir=str(model_dir))
    assert caught.value.code == "MODEL_LOAD"
    assert str(caught.value) == cli.stderr.strip()
    assert str(model_dir / dxnn) + ": .dxnn container v9 needs DX-RT >= 3.5.0" in str(caught.value)
    assert "use the v8 file (dxnn/2_4_0) or upgrade DX-RT to >= 3.5.0" in str(caught.value)
    assert "setup.sh" not in str(caught.value)


@pytest.mark.graph
@pytest.mark.skipif(platform.python_implementation() != "CPython",
                    reason="sys.getrefcount and gc.get_referrers are CPython reference counts")
def test_graph_error_type_holds_one_c_reference():
    """D2: the module keeps GraphError alive through exactly one reference
    no Python object can see - its own C pointer. Every other reference
    (the two modules' attributes, the type's own __mro__, ...) is one
    gc.get_referrers() lists. A second hidden reference is a leak. Run in a
    fresh interpreter so no test has raised, and kept, an instance."""
    import subprocess
    load_dx_graph()
    code = ("import gc, sys; sys.path.insert(0, {!r}); import dx_graph; "
            "E = dx_graph.GraphError; "
            "print(sys.getrefcount(E) - 1 - len(gc.get_referrers(E)))".format(PY_DIR))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


@pytest.mark.graph
def test_missing_model_matches_cli(tmp_path):
    """A registered model whose .dxnn is absent: same message as the CLI.

    The CLI prints nothing else on stderr for this case (the MODEL_MISSING
    line, the list, the download command), so the whole stripped stderr is
    compared."""
    dx_graph = load_dx_graph()
    graph = "{}/cascade_od_attr.json".format(GRAPHS)
    empty = tmp_path / "models"
    empty.mkdir()
    cli = run("multi_model_graph_sync", "--graph", graph, "--model-dir", str(empty),
              "--input", "sample/img/sample_people.jpg")
    assert cli.returncode == 1
    with pytest.raises(dx_graph.GraphError) as caught:
        dx_graph.Graph(str(PROJECT_ROOT / graph), model_dir=str(empty))
    assert caught.value.code == "MODEL_MISSING"
    assert str(caught.value) == cli.stderr.strip()
    assert isinstance(caught.value, ValueError)


@pytest.mark.graph
def test_a_graph_notes_old_names_and_aliases_on_stderr(tmp_path, capfd):
    """C5 deit ruling through dx_graph: building a graph prints the CLI run's
    alias notes on stderr, once per aliased node, before it fails on the
    empty model_dir; the GraphError text is the CLI's error without them."""
    dx_graph = load_dx_graph()
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({
        "version": 1, "name": "port",
        "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
                  {"id": "od", "model": "yolov8n"},
                  {"id": "cls", "model": "deit_base384_distilled"}],
        "edges": [{"from": "cam", "to": "od"}, {"from": "cam", "to": "cls"}]}), encoding="utf-8")
    empty = tmp_path / "models"
    empty.mkdir()
    capfd.readouterr()
    with pytest.raises(dx_graph.GraphError) as caught:
        dx_graph.Graph(str(path), model_dir=str(empty))
    assert caught.value.code == "MODEL_MISSING"
    err = capfd.readouterr().err
    assert err.splitlines() == ALIAS_NOTES, err
    assert "note: " not in str(caught.value)
    cli = run("multi_model_graph_sync", "--graph", str(path), "--model-dir", str(empty))
    assert cli.stderr.splitlines()[:2] == ALIAS_NOTES
    assert str(caught.value) == "\n".join(cli.stderr.splitlines()[2:])


@pytest.mark.graph
@pytest.mark.parametrize("kwargs", [
    {"executor": "fast"},
    {"max_frames_in_flight": 0},
    {"max_jobs_per_stage": -1},
    {"stall_timeout_ms": -1},
])
def test_bad_options_are_rejected_before_building(kwargs):
    dx_graph = load_dx_graph()
    with pytest.raises(ValueError):
        dx_graph.Graph(str(GRAPH_DIR / "cascade_od_attr.json"), **kwargs)


@pytest.mark.graph_e2e
@pytest.mark.parametrize("graph", RUNNABLE_SAMPLE_GRAPHS)
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_run_matches_cli(graph, executor, cli_single_image_report):
    require_artifacts(graph)
    import numpy
    dx_graph = load_dx_graph()
    frames = cli_single_image_report(graph, "multi_model_graph_" + executor)
    sources = graph_sources(graph)
    assert [f["index"] for f in frames] == [0] * len(sources)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        assert g.executor == executor
        assert g.streams == tuple(source for source, _ in sources)
        several = len(sources) > 1
        reports = [g.run(image, stream=source if several else None)
                   for source, image in sources]
    for (source, image), report, expected in zip(sources, reports, frames):
        assert report.frame_index == 0 and report.stream == source
        assert not report.error, report.error
        assert report.to_dict(arrays=False) == expected
        assert numpy.array_equal(report.frame, image)


@pytest.mark.graph_e2e
def test_frame_validation():
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    import numpy
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)
    height, width = image.shape[:2]
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="sync") as g:
        with pytest.raises(TypeError) as caught:
            g.run(image.astype(numpy.float32))
        assert str(caught.value) == FRAME_RULE + "float32 ({}, {}, 3)".format(height, width)
        with pytest.raises(TypeError) as caught:
            g.run(image.tolist())
        assert str(caught.value).startswith(FRAME_RULE)
        for bad in (image[:, :, 0],
                    numpy.zeros((height, width, 4), numpy.uint8),
                    numpy.zeros((0, width, 3), numpy.uint8),
                    numpy.zeros((height, 0, 3), numpy.uint8)):
            with pytest.raises(ValueError) as caught:
                g.run(bad)
            assert str(caught.value) == FRAME_RULE + "uint8 {}".format(bad.shape)

        view = image[:, ::2]
        assert not view.flags["C_CONTIGUOUS"]
        from_view = g.run(view)
        from_copy = g.run(numpy.ascontiguousarray(view))
        assert not from_view.error, from_view.error
        assert from_view.to_dict(arrays=False) == from_copy.to_dict(arrays=False)
        assert numpy.array_equal(from_view.frame, numpy.ascontiguousarray(view))


@pytest.mark.graph_e2e
def test_to_dict_arrays_match_summaries():
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    import numpy
    dx_graph = load_dx_graph()
    depths = {0: numpy.uint8, 1: numpy.int8, 2: numpy.uint16, 3: numpy.int16,
              4: numpy.int32, 5: numpy.float32, 6: numpy.float64}
    with dx_graph.Graph(str(GRAPH_DIR / graph)) as g:
        d = g.run(graph_source_image(graph)).to_dict()
    assert "labels_array" in d["nodes"]["seg"]["payload"]
    assert "values_array" in d["nodes"]["depth"]["payload"]
    pairs = find_arrays(d, [])
    for summary, array in pairs:
        assert array.shape[:2] == (summary["rows"], summary["cols"])
        channels = (summary["type"] >> 3) + 1
        assert array.ndim == (2 if channels == 1 else 3)
        assert array.dtype == depths[summary["type"] & 7]
        assert numpy.count_nonzero(array) == summary["nonzero"]
        assert summary["nonzero"] > 0


INSTANCE_GRAPH = (
    '{"version":1,"name":"inst","nodes":['
    '{"id":"cam","type":"source","uri":"sample/img/sample_people.jpg"},'
    '{"id":"seg","model":"yolov8n_seg"}],'
    '"edges":[{"from":"cam","to":"seg"}]}')


def mat_digest(array):
    """MatDigest (graph_consumer.cpp): FNV-1a 64 over the pixels, row by
    row, as 16 lower-case hex digits."""
    import numpy
    value = 1469598103934665603
    for byte in numpy.ascontiguousarray(array).tobytes():
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "{:016x}".format(value)


@pytest.fixture(scope="module")
def instance_cli_reports(tmp_path_factory):
    """(graph path, {"sync": frames, "async": frames}): the instance graph run
    once per CLI for the module, not once per executor parameter (U-45)."""
    tmp = tmp_path_factory.mktemp("instances")
    graph = tmp / "instances.json"
    graph.write_text(INSTANCE_GRAPH, encoding="utf-8")
    require_artifacts(str(graph))
    cli = {}
    for binary in ("sync", "async"):
        report = tmp / "{}.json".format(binary)
        result = run("multi_model_graph_" + binary, "--graph", str(graph), "--report", str(report))
        assert result.returncode == 0, result.stdout + result.stderr
        with open(str(report), encoding="utf-8") as handle:
            cli[binary] = json.load(handle)["frames"]
        assert len(cli[binary]) == 1
    assert cli["sync"] == cli["async"]
    return graph, cli


@pytest.mark.graph_e2e
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_instance_masks_match_cli(instance_cli_reports, executor):
    """C4: an instance-segmentation graph on the NPU. to_dict(arrays=False)
    equals the CLI's --report for the same executor (and the two CLIs agree),
    and every item's mask_array is exactly the mask its summary describes -
    rows, cols, type, nonzero and digest."""
    graph, cli = instance_cli_reports
    import numpy
    dx_graph = load_dx_graph()

    with dx_graph.Graph(str(graph), executor=executor) as g:
        report = g.run(graph_source_image(str(graph)))
    assert not report.error, report.error
    assert report.to_dict(arrays=False) == cli[executor][0]

    d = report.to_dict()
    items = d["nodes"]["seg"]["payload"]["items"]
    assert d["nodes"]["seg"]["payload"]["shape"] == "instances"
    assert len(items) >= 2  # not vacuous: people in the picture
    depths = {0: numpy.uint8, 1: numpy.int8, 2: numpy.uint16, 3: numpy.int16,
              4: numpy.int32, 5: numpy.float32, 6: numpy.float64}
    for item in items:
        assert "mask" in item
        summary, array = item["mask"], item["mask_array"]
        channels = (summary["type"] >> 3) + 1
        assert array.shape[:2] == (summary["rows"], summary["cols"])
        assert array.ndim == (2 if channels == 1 else 3)
        assert array.dtype == depths[summary["type"] & 7]
        assert numpy.count_nonzero(array) == summary["nonzero"] > 0
        assert mat_digest(array) == summary["digest"]


@pytest.mark.graph_e2e
def test_concurrent_use_is_refused():
    import threading
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as g:
        holder = threading.Thread(target=g._lock.acquire)
        holder.start()
        holder.join()
        with pytest.raises(RuntimeError) as caught:
            g.run(image)
        assert str(caught.value) == "dx_graph.Graph is already running in another thread"
        g._lock.release()
        assert not g.run(image).error


@pytest.mark.graph_e2e
def test_closed_graph_refuses():
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)
    g = dx_graph.Graph(str(GRAPH_DIR / graph))
    assert g.options == {"max_frames_in_flight": 16, "max_jobs_per_stage": 0,
                         "stall_timeout_ms": 0}
    g.close()
    g.close()  # idempotent
    with pytest.raises(RuntimeError) as caught:
        g.run(image)
    assert str(caught.value) == "dx_graph.Graph is closed"
    with dx_graph.Graph(str(GRAPH_DIR / graph)) as h:
        pass
    with pytest.raises(RuntimeError):
        h.run(image)


@pytest.mark.graph_e2e
def test_options_are_the_async_executor_s_only():
    """Minor #8: the three options configure the async executor only. A
    sync Graph still validates them, but its `options` is empty rather than
    showing values nothing uses."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    given = {"max_frames_in_flight": 4, "max_jobs_per_stage": 2, "stall_timeout_ms": 5000}
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="sync", **given) as g:
        assert dict(g.options) == {}
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async", **given) as g:
        assert dict(g.options) == given
    with pytest.raises(ValueError):  # validated all the same
        dx_graph.Graph(str(GRAPH_DIR / graph), executor="sync", max_frames_in_flight=0)


@pytest.mark.graph_e2e
def test_a_model_the_npu_cannot_hold_raises_model_load(tmp_path):
    require_artifacts("handoff_sr_od_cls.json")
    dx_graph = load_dx_graph()
    path = tmp_path / "two_sr.json"
    path.write_text(json.dumps(TWO_SR_GRAPH))
    try:
        graph = dx_graph.Graph(str(path))
    except dx_graph.GraphError as error:
        assert error.code == "MODEL_LOAD"
        cli = run("multi_model_graph_sync", "--graph", str(path))
        assert cli.returncode == 1
        assert str(error) in cli.stderr  # the CLI's own message, word for word
        return
    graph.close()
    pytest.skip("this NPU holds a second realesrgan_x2 next to yolov8n and resnet50")


@pytest.mark.graph_e2e
def test_interrupt_during_run_is_raised_and_graph_stays_usable():
    """P12: a signal that arrives while the engine runs (GIL released) is
    caught by the interrupt hook, kept, and raised once the call returns -
    not swallowed - and the Graph runs the next frame normally. A custom
    handler stands in for KeyboardInterrupt so a stray delivery cannot abort
    the pytest session."""
    import os
    import signal
    import threading
    import time
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)

    class Interrupted(Exception):
        pass

    def handler(signum, frame):
        raise Interrupted()

    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as g:
        previous = signal.signal(signal.SIGINT, handler)
        timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
        try:
            with pytest.raises(Interrupted):
                timer.start()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    g.run(image)
        finally:
            timer.cancel()
            signal.signal(signal.SIGINT, previous)
        report = g.run(image)
        assert report.frame_index == 0 and not report.error


# The interrupt hook against a stage that never completes. No hardware:
# _dx_graph._stalled_graph_for_tests builds a real _Graph - the same hooks,
# the same async executor, the same CallEngine - over one node whose stage
# drops every job, so a run waits, quiet, until stall_timeout_ms throws.

STALL_FRAME_SHAPE = (48, 64, 3)
INTERRUPT_CHECK_MS = 50  # the hook's throttle window (D1)
STUCK_LINE = "interrupted: still waiting on stage(s): stuck\n"


def stalled_graph(dx_graph, stall_timeout_ms):
    return graph_on(dx_graph, dx_graph._dx_graph._stalled_graph_for_tests(stall_timeout_ms))


@pytest.mark.graph
def test_interrupt_hook_checks_signals_at_most_once_per_window(capfd):
    """D1: while nothing completes, the executor calls the hook every 5 ms
    wait slice; the hook takes the GIL and checks for signals at most once
    per 50 ms window, and still does so throughout the wait. With no signal
    the run ends in the executor's own stall error and nothing on stderr."""
    import time
    import numpy
    dx_graph = load_dx_graph()
    g = stalled_graph(dx_graph, 600)
    core = g._core
    start = time.monotonic()
    with pytest.raises(RuntimeError) as caught:
        g.run(numpy.zeros(STALL_FRAME_SHAPE, numpy.uint8))
    elapsed_ms = (time.monotonic() - start) * 1000
    assert str(caught.value) == ("AsyncExecutor stalled: no completion for 600 ms; "
                                 "waiting on \"stuck\"")
    checks = core._interrupt_checks
    # The first check comes at once, then one per window. The lower bound
    # (a window of at most 75 ms) fails a 100 ms throttle; the upper bound
    # fails no throttle at all.
    assert elapsed_ms / 75 + 1 <= checks <= elapsed_ms / INTERRUPT_CHECK_MS + 2, \
        (checks, elapsed_ms)
    assert capfd.readouterr().err == ""
    g.close()


@pytest.mark.graph
def test_ctrl_c_during_a_stalled_wait_is_honoured(capfd):
    """D1 + D5: a SIGINT during a wait on a stage that never completes runs
    the Python handler from inside the engine call - within the throttle
    window plus 100 ms of the signal, not when the call returns - and the
    hook, now answering "interrupted", makes the executor name the stuck
    stage on stderr once it has been quiet for 1 s (the CLI's own line).
    The stall then ends the call, and what is raised is the kept
    interrupt, not the stall error. A custom handler stands in for
    KeyboardInterrupt so a stray delivery cannot abort the pytest session."""
    import os
    import signal
    import threading
    import time
    import numpy
    dx_graph = load_dx_graph()
    g = stalled_graph(dx_graph, 1500)
    seen = {}

    class Interrupted(Exception):
        pass

    def handler(signum, frame):
        seen["handled"] = time.monotonic()
        raise Interrupted()

    def interrupt():
        seen["sent"] = time.monotonic()
        os.kill(os.getpid(), signal.SIGINT)

    previous = signal.signal(signal.SIGINT, handler)
    timer = threading.Timer(0.3, interrupt)
    try:
        with pytest.raises(Interrupted):
            timer.start()
            started = time.monotonic()
            g.run(numpy.zeros(STALL_FRAME_SHAPE, numpy.uint8))
        returned = time.monotonic()
    finally:
        timer.cancel()
        signal.signal(signal.SIGINT, previous)
    latency = seen["handled"] - seen["sent"]
    assert latency < 2 * INTERRUPT_CHECK_MS / 1000.0, latency
    # The checks up to the one that ran the handler: the first at once, then
    # one per window of at most 75 ms (a 100 ms throttle makes too few).
    until_handled_ms = (seen["handled"] - started) * 1000
    checks = g._core._interrupt_checks
    assert checks >= until_handled_ms / 75 + 1, (checks, until_handled_ms)
    assert returned - seen["sent"] > 1.0  # handled inside the call, not after it
    assert capfd.readouterr().err == STUCK_LINE
    g.close()


@pytest.mark.graph
def test_stalled_graph_refuses_a_zero_stall_timeout():
    """0 means wait forever, and the stalled graph never completes: refused
    at once instead of building a graph that would hang its caller."""
    dx_graph = load_dx_graph()
    with pytest.raises(ValueError) as caught:
        dx_graph._dx_graph._stalled_graph_for_tests(0)
    assert "stall_timeout_ms must be > 0" in str(caught.value)


@pytest.mark.graph
def test_a_closed_core_refuses_stream_ids():
    """A closed core has no graph to name the streams of: stream_ids says
    so, like every other call, instead of reading a graph that is gone."""
    dx_graph = load_dx_graph()
    core = dx_graph._dx_graph._stalled_graph_for_tests(100)
    assert list(core.stream_ids) == ["cam"]
    core.close()
    with pytest.raises(RuntimeError) as caught:
        core.stream_ids
    assert str(caught.value) == "dx_graph.Graph is closed"


@pytest.mark.graph
def test_to_dict_keeps_the_cli_spelling_of_non_finite_numbers():
    """U-11: to_dict() is json.loads of the CLI serializer's own text, so a
    NaN is the string "NaN" in both - never converted, never null."""
    dx_graph = load_dx_graph()

    class FakeCore(object):
        def json(self):
            return ('{"nodes": {"emb": {"payload": {"shape": "vector", '
                    '"values": ["NaN", "Infinity", "-Infinity", 0.5]}}}}')

        def arrays(self):
            return []

    d = dx_graph.Report(FakeCore()).to_dict()
    assert d["nodes"]["emb"]["payload"]["values"] == ["NaN", "Infinity", "-Infinity", 0.5]


# ------------------------------------------------------------------ stream
#
# stream() is pipelined: under the async executor the report handed out is
# not for the frame just submitted. Every test below would pass on a
# one-frame-at-a-time implementation except the ones that pin exactly that.

IN_FLIGHT = "dx_graph.Graph has reports in flight; finish the stream first"
SAME_THREAD = ("dx_graph.Graph is already in use by this thread "
               "(e.g. run() inside a stream loop)")


def cli_video_report(tmp_path, graph, executor, video):
    report = tmp_path / "{}_{}.json".format(executor, graph)
    result = run("multi_model_graph_" + executor, "--graph", "{}/{}".format(GRAPHS, graph),
                 "--input", str(video), "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    with open(str(report), encoding="utf-8") as handle:
        return json.load(handle)["frames"]


@pytest.mark.graph_parity
@pytest.mark.parametrize("graph", ["cascade_od_reid_track.json", "handoff_denoise_od.json"])
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_stream_matches_cli_on_a_moving_video(tmp_path, graph, executor):
    """The graph reads the video itself (source=...): the same reports as the
    CLI's --input/--report, in frame order, the tail included."""
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = write_moving_video(tmp_path / "moving.avi")
    cli = cli_video_report(tmp_path, graph, executor, video)
    assert [f["index"] for f in cli] == list(range(8))
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        reports = [r.to_dict(arrays=False) for r in g.stream(source=str(video))]
    assert [r["index"] for r in reports] == list(range(8))
    assert reports == cli


@pytest.mark.graph_parity
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_reports_kept_past_the_next_read_keep_their_own_frame(tmp_path, executor):
    """A Report outlives the source's next read: every report of a stream
    kept in a list, then turned into dicts after the loop, is what the same
    reports give inside the loop and what the CLI writes. The sync
    executor's report holds the source frame without a copy, so a read
    buffer the source reused would carry the newest frame into every
    earlier report."""
    graph = "cascade_od_reid_track.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = write_moving_video(tmp_path / "moving.avi")
    cli = cli_video_report(tmp_path, graph, executor, video)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        in_loop = [r.to_dict(arrays=False) for r in g.stream(source=str(video))]
        reports = list(g.stream(source=str(video)))
    after = [r.to_dict(arrays=False) for r in reports]
    assert [r["index"] for r in after] == list(range(8))
    assert after == in_loop
    assert after == cli


@pytest.mark.graph_parity
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_a_second_stream_starts_on_fresh_trackers(tmp_path, executor):
    """I1: the tracker lives in the graph, so a stream that did not reset it
    would carry the first stream's tracks into the second and hand out other
    ids. Both streams through ONE Graph must equal the CLI's --report."""
    graph = "cascade_od_reid_track.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = write_moving_video(tmp_path / "moving.avi")
    cli = cli_video_report(tmp_path, graph, executor, video)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        first = [r.to_dict(arrays=False) for r in g.stream(source=str(video))]
        second = [r.to_dict(arrays=False) for r in g.stream(source=str(video))]
    assert first == cli
    assert second == cli


@pytest.mark.graph_parity
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_run_after_a_stream_equals_a_fresh_graph(tmp_path, executor):
    """I1: run() is one independent image, as the CLI's single-image run:
    after a stream (and after another run) it gives what a fresh Graph's
    run() gives, track ids included."""
    graph = "cascade_od_reid_track.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = write_moving_video(tmp_path / "moving.avi")
    image = graph_source_image(graph)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as fresh:
        expected = fresh.run(image).to_dict(arrays=False)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        assert len(list(g.stream(source=str(video)))) == 8
        after_stream = g.run(image).to_dict(arrays=False)
        after_run = g.run(image).to_dict(arrays=False)
    assert after_stream == expected
    assert after_run == expected


@pytest.mark.graph_parity
def test_stream_of_numpy_frames_async_equals_sync():
    """numpy frames through a tracker graph: the pipelined executor with 4
    frames in flight must hand out exactly the one-at-a-time reports."""
    graph = "cascade_od_reid_track.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    frames = shifted_sample_frames(6)
    results = {}
    for executor, options in (("sync", {}), ("async", {"max_frames_in_flight": 4})):
        with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor, **options) as g:
            results[executor] = [r.to_dict(arrays=False) for r in g.stream(frames)]
    assert [r["index"] for r in results["async"]] == list(range(6))
    assert all(not r["error"] for r in results["async"]), results["async"]
    assert results["async"] == results["sync"]


@pytest.mark.graph_e2e
def test_stream_early_break_leaves_graph_reusable(tmp_path):
    """P14: leaving a stream early - break, an exception in the loop body -
    finishes the frames in flight and discards their reports, so the next
    run() and stream() are handed no leftover report and are numbered from
    0. fanout_od_seg_depth has no tracker: this pins the in-flight cleanup
    only; tracker state across calls is pinned by
    test_a_second_stream_starts_on_fresh_trackers and
    test_run_after_a_stream_equals_a_fresh_graph. With 8 frames and 16 in
    flight the first report only comes out after the source ends, so the
    break lands in the tail drain."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = str(write_moving_video(tmp_path / "moving.avi"))
    image = graph_source_image(graph)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as fresh:
        expected = fresh.run(image).to_dict(arrays=False)

    class Boom(Exception):
        pass

    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as g:
        for r in g.stream(source=video):
            assert r.frame_index == 0
            break
        assert g.run(image).to_dict(arrays=False) == expected

        with pytest.raises(Boom):
            for r in g.stream(shifted_sample_frames(4)):
                raise Boom()
        assert g.run(image).to_dict(arrays=False) == expected

        assert [r.frame_index for r in g.stream(source=video, limit=3)] == [0, 1, 2]
        assert [r.frame_index for r in g.stream(source=video)] == list(range(8))
        assert g.run(image).to_dict(arrays=False) == expected


@pytest.mark.graph_e2e
def test_stream_rejects_frames_and_source_together(tmp_path):
    """Refused when stream() is called, not at the first next()."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)
    with dx_graph.Graph(str(GRAPH_DIR / graph)) as g:
        with pytest.raises(ValueError) as caught:
            g.stream([image], source=str(GRAPH_DIR / graph))
        assert str(caught.value) == "pass frames or source, not both"
        with pytest.raises(ValueError):
            g.stream([image], limit=-1)
        assert not g.run(image).error  # nothing was left holding the lock


@pytest.mark.graph_e2e
def test_stream_reports_carry_their_own_frame(tmp_path):
    """P7: under pipelining each report hands back the frame it was computed
    on - not the frame most recently read."""
    import cv2
    import numpy
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = str(write_moving_video(tmp_path / "moving.avi"))
    capture = cv2.VideoCapture(video)
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert len(decoded) == 8
    assert not numpy.array_equal(decoded[0], decoded[7])
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as g:
        reports = list(g.stream(source=video))
    assert [r.frame_index for r in reports] == list(range(8))
    for r in reports:
        assert numpy.array_equal(r.frame, decoded[r.frame_index]), r.frame_index


@pytest.mark.graph_e2e
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_run_refuses_while_reports_are_in_flight(executor):
    """Ruling R2(a): run() never returns a report left over from submitted
    frames; it refuses until they are finished and taken."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        expected = g.run(image).to_dict(arrays=False)
        core = g._core
        core.submit(image[:, ::-1])  # a different frame, so a stale report would show
        with pytest.raises(RuntimeError) as caught:
            g.run(image)
        assert str(caught.value) == IN_FLIGHT
        core.finish()
        with pytest.raises(RuntimeError) as caught:
            g.run(image)  # finished, but its report is still untaken
        assert str(caught.value) == IN_FLIGHT
        taken = core.try_next()
        assert taken is not None and core.try_next() is None
        assert g.run(image).to_dict(arrays=False) == expected


@pytest.mark.graph_e2e
def test_run_inside_a_stream_is_refused_with_its_own_message():
    """Ruling R2(b): same-thread re-entry is named as such, not blamed on
    another thread; the stream itself is unaffected."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    image = graph_source_image(graph)
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as g:
        seen = []
        for r in g.stream([image, image]):
            with pytest.raises(RuntimeError) as caught:
                g.run(image)
            assert str(caught.value) == SAME_THREAD
            with pytest.raises(RuntimeError) as caught:
                next(iter(g.stream([image])))
            assert str(caught.value) == SAME_THREAD
            seen.append(r.frame_index)
        assert seen == [0, 1]
        assert not g.run(image).error


@pytest.mark.graph_e2e
def test_stream_source_errors_match_cli(tmp_path):
    """stream(source=...) opens the input by the CLI's own rule
    (consumer::OpenGraphSource), so an unopenable path and a video with no
    frames are the CLI's GRAPH_SCHEMA errors, word for word. OpenCV's own
    warnings on stderr come before the CLI's line and are not compared."""
    import cv2
    graph = "handoff_denoise_od.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    empty = tmp_path / "zero.avi"
    writer = cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    writer.release()
    with dx_graph.Graph(str(GRAPH_DIR / graph)) as g:
        for uri, what in (("nope/missing.jpg", "Failed to open"),
                          (str(empty), "produced no frames")):
            cli = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
                      "--input", uri)
            assert cli.returncode == 1
            expected = cli.stderr[cli.stderr.index("ERROR [GRAPH_SCHEMA]"):].strip()
            with pytest.raises(dx_graph.GraphError) as caught:
                list(g.stream(source=uri))
            assert caught.value.code == "GRAPH_SCHEMA"
            assert what in expected, expected
            assert str(caught.value) == expected
        assert not g.run(graph_source_image(graph)).error


# --------------------------------------------------------- several sources


@pytest.mark.graph_parity
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_stream_of_two_sources_matches_cli(tmp_path, executor):
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    cam1, cam2 = write_two_stream_videos(tmp_path)
    report = tmp_path / "cli.json"
    result = run("multi_model_graph_" + executor, "--graph", "{}/{}".format(GRAPHS, MULTISTREAM),
                 "--input", "cam1={}".format(cam1), "--input", "cam2={}".format(cam2),
                 "--report", str(report))
    assert result.returncode == 0, result.stdout + result.stderr
    with open(str(report), encoding="utf-8") as handle:
        cli = json.load(handle)["frames"]
    with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM), executor=executor) as g:
        assert g.streams == ("cam1", "cam2")
        reports = list(g.stream(sources={"cam1": str(cam1), "cam2": str(cam2)}))
    assert [(r.stream, r.frame_index) for r in reports] == \
        read_in_turn([("cam1", 8), ("cam2", 5)])
    assert [r.to_dict(arrays=False) for r in reports] == cli


@pytest.mark.graph_parity
def test_stream_of_source_frame_pairs_async_equals_sync():
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    first = shifted_sample_frames(5)
    second = shifted_sample_frames(3, image="sample_person_a1.jpg")
    pairs = []
    for k in range(5):
        pairs.append(("cam1", first[k]))
        if k < 3:
            pairs.append(("cam2", second[k]))
    results = {}
    for executor, options in (("sync", {}), ("async", {"max_frames_in_flight": 4})):
        with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM), executor=executor, **options) as g:
            results[executor] = [r.to_dict(arrays=False) for r in g.stream(pairs)]
    assert [(r["stream"], r["index"]) for r in results["async"]] == \
        read_in_turn([("cam1", 5), ("cam2", 3)])
    assert all(not r["error"] for r in results["async"]), results["async"]
    assert results["async"] == results["sync"]


@pytest.mark.graph_e2e
def test_sources_limit_counts_each_stream(tmp_path):
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    cam1, cam2 = write_two_stream_videos(tmp_path)
    with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM)) as g:
        got = [(r.stream, r.frame_index)
               for r in g.stream(sources={"cam1": str(cam1), "cam2": str(cam2)}, limit=3)]
        own = [(r.stream, r.frame_index) for r in g.stream()]  # the graph's own images
    assert got == read_in_turn([("cam1", 3), ("cam2", 3)])
    assert own == [("cam1", 0), ("cam2", 0)]


@pytest.mark.graph_e2e
def test_run_names_the_stream(tmp_path):
    import cv2
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    image = cv2.imread(str(PROJECT_ROOT / "sample" / "img" / "sample_person_a1.jpg"))
    with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM)) as g:
        report = g.run(image, stream="cam2")
    assert report.stream == "cam2" and report.frame_index == 0
    d = report.to_dict(arrays=False)
    assert d["stream"] == "cam2" and "cam2" in d["nodes"] and "cam1" not in d["nodes"]


def write_empty_video(path):
    """A video file with no frame in it."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    writer.release()
    return path


def cli_no_frames_lines(executor, inputs):
    """The CLI's "produced no frames" lines for these --input values, as one
    text: what a GraphError of stream(sources=...) must say, word for word."""
    args = ["--graph", "{}/{}".format(GRAPHS, MULTISTREAM)]
    for value in inputs:
        args += ["--input", value]
    cli = run("multi_model_graph_" + executor, *args)
    assert cli.returncode == 1, cli.stdout + cli.stderr
    return cli.stderr[cli.stderr.index("ERROR [GRAPH_SCHEMA]"):].strip()


@pytest.mark.graph_e2e
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_an_empty_source_is_raised_after_the_other_streams_reports(tmp_path, executor):
    """Spec R9, as the CLI: an empty stream does not stop the others. Every
    report of the other streams comes out first; then, at the normal end,
    one GraphError names the empty stream with the CLI's message."""
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    _, five = write_two_stream_videos(tmp_path)  # 5 frames
    empty = write_empty_video(tmp_path / "zero.avi")
    expected = cli_no_frames_lines(executor, ["cam1={}".format(five), "cam2={}".format(empty)])
    with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM), executor=executor) as g:
        got = []
        with pytest.raises(dx_graph.GraphError) as caught:
            for r in g.stream(sources={"cam1": str(five), "cam2": str(empty)}):
                assert not r.error, r.error
                got.append((r.stream, r.frame_index))
        assert got == [("cam1", k) for k in range(5)]
        assert caught.value.code == "GRAPH_SCHEMA"
        assert str(caught.value) == expected
        assert str(empty) in expected and "produced no frames" in expected
        assert not g._lock.locked()
        assert [r.stream for r in g.stream()] == ["cam1", "cam2"]  # reusable at once


@pytest.mark.graph_e2e
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_two_empty_sources_are_both_named_in_one_error(tmp_path, executor):
    """Spec R9: every empty stream is named, in declaration order, in the one
    GraphError - the CLI's lines joined by newlines."""
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    first = write_empty_video(tmp_path / "zero1.avi")
    second = write_empty_video(tmp_path / "zero2.avi")
    expected = cli_no_frames_lines(executor, ["cam1={}".format(first), "cam2={}".format(second)])
    with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM), executor=executor) as g:
        with pytest.raises(dx_graph.GraphError) as caught:
            list(g.stream(sources={"cam1": str(first), "cam2": str(second)}))
    assert caught.value.code == "GRAPH_SCHEMA"
    assert str(caught.value) == expected
    lines = [line for line in str(caught.value).splitlines() if line.startswith("ERROR")]
    assert lines == ['ERROR [GRAPH_SCHEMA] source "{}": produced no frames'.format(p)
                     for p in (first, second)]


@pytest.mark.graph_e2e
def test_breaking_out_of_a_stream_with_an_empty_source_raises_nothing(tmp_path):
    """The empty-stream error belongs to a stream that ran to its end: one
    left early with break is abandoned quietly, as any other."""
    require_artifacts(MULTISTREAM)
    dx_graph = load_dx_graph()
    _, five = write_two_stream_videos(tmp_path)
    empty = write_empty_video(tmp_path / "zero.avi")
    with dx_graph.Graph(str(GRAPH_DIR / MULTISTREAM)) as g:
        for r in g.stream(sources={"cam1": str(five), "cam2": str(empty)}):
            assert r.stream == "cam1"
            break
        assert not g._lock.locked()


# ------------------------------------------------------------------ render


def read_ppm_as_bgr(path):
    """A binary PPM (P6, maxval 255) -> numpy (rows, cols, 3) uint8 in BGR.

    PPM stores RGB; cv::imwrite converted from BGR to write it, so the
    channels are reversed back. Header tokens are whitespace-separated and
    may be interleaved with # comments; exactly one whitespace byte follows
    the maxval."""
    import numpy
    with open(str(path), "rb") as handle:
        data = handle.read()
    tokens = []
    pos = 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1
    assert tokens[0] == b"P6" and tokens[3] == b"255", tokens
    cols, rows = int(tokens[1]), int(tokens[2])
    assert len(data) == pos + rows * cols * 3, path
    rgb = numpy.frombuffer(data, numpy.uint8, rows * cols * 3, pos).reshape(rows, cols, 3)
    return numpy.ascontiguousarray(rgb[:, :, ::-1])


@pytest.mark.graph_parity
def test_render_matches_cli_ppm(tmp_path):
    """P8 / criterion 3: render(report) is the CLI's --output, pixel for
    pixel, and uses the report's OWN frame. Under the async executor all 8
    frames are submitted before the first report comes out, so rendering
    onto "the latest frame" (or a blank canvas) would differ on every frame
    but the last - the moving video makes each frame distinct. A single image
    writes exactly --output, no number."""
    import numpy
    graph = "fanout_od_seg_pose.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    video = write_moving_video(tmp_path / "moving.avi")
    out = tmp_path / "video" / "out.ppm"
    out.parent.mkdir()
    cli = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
              "--input", str(video), "--output", str(out))
    assert cli.returncode == 0, cli.stdout + cli.stderr
    written = sorted(p.name for p in out.parent.iterdir())
    assert written == ["out_{:06d}.ppm".format(k) for k in range(8)], written

    image = graph_source_image(graph)
    single = tmp_path / "image" / "out.ppm"
    single.parent.mkdir()
    cli = run("multi_model_graph_async", "--graph", "{}/{}".format(GRAPHS, graph),
              "--output", str(single))
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert sorted(p.name for p in single.parent.iterdir()) == ["out.ppm"]

    with dx_graph.Graph(str(GRAPH_DIR / graph), executor="async") as g:
        seen = []
        for r in g.stream(source=str(video)):
            assert not r.error, r.error
            canvas = g.render(r)  # inside the loop: render is not "re-entry"
            expected = read_ppm_as_bgr(out.parent / "out_{:06d}.ppm".format(r.frame_index))
            assert canvas.dtype == numpy.uint8 and canvas.shape == expected.shape
            assert numpy.array_equal(canvas, expected), r.frame_index
            assert not numpy.array_equal(canvas, r.frame)  # something was drawn
            seen.append(r.frame_index)
        assert seen == list(range(8))

        report = g.run(image)
        assert numpy.array_equal(g.render(report), read_ppm_as_bgr(single))
        with pytest.raises(TypeError):
            g.render(report._r)
    with pytest.raises(RuntimeError) as caught:
        g.render(report)
    assert str(caught.value) == "dx_graph.Graph is closed"


# ------------------------------------------ stream edges (ruling R3)


class CountingFrames(object):
    """An iterator over `frames` that records how many items were pulled."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pulled >= len(self._frames):
            raise StopIteration
        frame = self._frames[self.pulled]
        self.pulled += 1
        return frame


@pytest.mark.graph_e2e
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_stream_limit_does_not_pull_past_the_limit(executor):
    """R3(a): like the CLI's --frames, the limit is checked after each
    submit, so the (N+1)-th item of the caller's iterator is never taken."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    frames = CountingFrames(shifted_sample_frames(5))
    with dx_graph.Graph(str(GRAPH_DIR / graph), executor=executor) as g:
        assert [r.frame_index for r in g.stream(frames, limit=3)] == [0, 1, 2]
        assert frames.pulled == 3
        rest = [r.frame_index for r in g.stream(frames)]
        assert rest == [0, 1] and frames.pulled == 5


@pytest.mark.graph_e2e
def test_close_during_a_suspended_stream_ends_it(tmp_path):
    """R3(b): close() while this thread's stream is suspended at a report:
    the stream's next step ends it - no further reads or submits - and what
    was in flight is finished and dropped, then the models are released."""
    graph = "fanout_od_seg_depth.json"
    require_artifacts(graph)
    dx_graph = load_dx_graph()
    frames = CountingFrames(shifted_sample_frames(8))
    g = dx_graph.Graph(str(GRAPH_DIR / graph), executor="async", max_frames_in_flight=2)
    it = g.stream(frames)
    first = next(it)
    assert first.frame_index == 0
    pulled = frames.pulled
    assert pulled < 8
    core = g._core
    g.close()
    assert not core.closed  # the suspended stream still holds it
    assert list(it) == []
    assert frames.pulled == pulled
    assert core.closed
    with pytest.raises(RuntimeError) as caught:
        g.run(first.frame)
    assert str(caught.value) == "dx_graph.Graph is closed"


class FakeCore(object):
    """Stands in for _dx_graph._Graph: reports appear only after finish(),
    as when every frame is still in flight when the input ends. Records the
    call order."""

    def __init__(self):
        self.calls = []
        self._ready = []
        self._submitted = 0

    def submit(self, frame):
        self.calls.append("submit")
        self._submitted += 1

    def try_next(self):
        self.calls.append("try_next")
        return self._ready.pop(0) if self._ready else None

    def finish(self):
        self.calls.append("finish")
        self._ready = ["r{}".format(k) for k in range(self._submitted)]
        self._submitted = 0

    def abandon(self):
        self.calls.append("abandon")
        self._ready = []
        self._submitted = 0

    def close(self):
        self.calls.append("close")


def graph_on(dx_graph, core):
    """A Graph wired to `core` without building anything."""
    import threading
    g = object.__new__(dx_graph.Graph)
    g._lock = threading.Lock()
    g._owner = None
    g._core = core
    g._executor = "async"
    g._streams = ("cam",)
    return g


@pytest.mark.graph
def test_break_in_the_tail_drain_abandons_after_finish():
    """R3(d): no hardware, deterministic. Every report comes out of the tail
    drain (after finish()); a break there must still abandon() what is left,
    and a stream consumed to the end must not.

    The first case relies on CPython reference counting: the unnamed
    generator of `for r in g.stream(...)` is closed - its finally clause
    run - the moment the loop is left, so the calls are complete on the
    next line. On an interpreter that collects later (PyPy) the case would
    have to name the iterator and call its close() before checking."""
    dx_graph = load_dx_graph()
    core = FakeCore()
    g = graph_on(dx_graph, core)
    for r in g.stream([1, 2, 3]):
        assert r._r == "r0"
        break
    assert core.calls == ["submit", "try_next"] * 3 + ["finish", "try_next", "abandon"]
    assert not g._lock.locked()

    core.calls[:] = []
    assert [r._r for r in g.stream([1, 2])] == ["r0", "r1"]
    assert core.calls == ["submit", "try_next"] * 2 + ["finish"] + ["try_next"] * 3
    assert not g._lock.locked()

    core.calls[:] = []
    it = g.stream([1, 2, 3])
    assert next(it)._r == "r0"
    g.close()
    assert list(it) == []
    assert core.calls == (["submit", "try_next"] * 3 +
                          ["finish", "try_next", "abandon", "close"])
    assert not g._lock.locked()


@pytest.mark.graph
def test_close_racing_a_call_on_another_thread_says_closed():
    """D3: no hardware, deterministic. A call on another thread holds the
    Graph; close() lands between a new call's closed check and its lock
    attempt. The new call is refused as "closed" - what the Graph now is -
    not as "running in another thread", and close() leaves the core to the
    call that holds it."""
    dx_graph = load_dx_graph()
    core = FakeCore()
    g = graph_on(dx_graph, core)
    held = g._lock
    held.acquire()  # the call running on the other thread

    class CloseLandsFirst(object):
        """g's lock, with close() running just before the first attempt."""

        def __init__(self):
            self.raced = False

        def acquire(self, blocking=True):
            if not self.raced:
                self.raced = True
                g.close()
            return held.acquire(blocking)

        def release(self):
            held.release()

    g._lock = CloseLandsFirst()
    try:
        with pytest.raises(RuntimeError) as caught:
            g.run(None)
        assert str(caught.value) == "dx_graph.Graph is closed"
        assert g._lock.raced
        assert "close" not in core.calls
    finally:
        held.release()


# -------------------------------------------- several sources, no hardware


class FakeTwoSourceCore(FakeCore):
    """FakeCore for a graph with sources cam1 and cam2: submit() records the
    stream each frame was given."""

    def submit(self, frame, stream=0):
        self.calls.append(("submit", stream))
        self._submitted += 1


def two_source_graph_on(dx_graph, core):
    g = graph_on(dx_graph, core)
    g._streams = ("cam1", "cam2")
    return g


@pytest.mark.graph
def test_a_two_source_graph_refuses_calls_that_name_no_stream():
    dx_graph = load_dx_graph()
    g = two_source_graph_on(dx_graph, FakeTwoSourceCore())
    with pytest.raises(ValueError) as caught:
        g.stream(source="clip.mp4")
    assert str(caught.value) == (
        "source= replaces the one source of a single-source graph, and this graph "
        "has 2 (cam1, cam2): pass sources={\"<id>\": uri, ...}")
    with pytest.raises(ValueError) as caught:
        g.stream(sources={"camX": "clip.mp4"})
    assert str(caught.value) == "'camX' is not a source of this graph; its sources: cam1, cam2"
    with pytest.raises(ValueError) as caught:
        g.stream([("cam1", 1)], limit=2)
    assert str(caught.value) == (
        "limit counts each stream's frames when the graph reads its sources; with "
        "frames= on a graph with several sources, pass only the frames you want")
    with pytest.raises(ValueError) as caught:
        g.stream([], sources={})
    assert str(caught.value) == "pass frames or sources, not both"
    with pytest.raises(ValueError) as caught:
        g.stream(source="a", sources={})
    assert str(caught.value) == "pass source or sources, not both"
    with pytest.raises(TypeError) as caught:
        g.stream(sources=["cam1"])
    assert str(caught.value) == "sources must be a mapping of source id to uri, got list"
    with pytest.raises(ValueError) as caught:
        g.run(object())
    assert str(caught.value) == (
        "run() on a graph with 2 sources (cam1, cam2) needs stream=: the id of the "
        "source this frame belongs to")
    with pytest.raises(ValueError) as caught:
        g.run(object(), stream="camX")
    assert str(caught.value) == "'camX' is not a source of this graph; its sources: cam1, cam2"
    assert not g._lock.locked()


@pytest.mark.graph
def test_a_two_source_graph_submits_each_pair_to_its_stream():
    dx_graph = load_dx_graph()
    core = FakeTwoSourceCore()
    g = two_source_graph_on(dx_graph, core)
    assert [r._r for r in g.stream([("cam2", "a"), ("cam1", "b")])] == ["r0", "r1"]
    assert core.calls == [("submit", 1), "try_next", ("submit", 0), "try_next",
                          "finish", "try_next", "try_next", "try_next"]
    with pytest.raises(TypeError) as caught:
        list(g.stream(["not a pair"]))
    assert "(source_id, frame) pairs" in str(caught.value)
    with pytest.raises(ValueError):
        list(g.stream([("camX", "a")]))
    assert not g._lock.locked()


# ----------------------------------------------------------------- example


class ExampleReport(object):
    def __init__(self, frame_index, error, stream="cam"):
        self.frame_index = frame_index
        self.error = error
        self.stream = stream

    def to_dict(self, arrays=True):
        return {"index": self.frame_index, "error": self.error, "nodes": {"od": {}}}


class ExampleGraph(object):
    """Stands in for dx_graph.Graph in examples/run_graph.py: frame 1 fails."""

    streams = ("cam",)

    def __init__(self, path, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def stream(self, source=None, limit=None):
        return iter([ExampleReport(0, ""), ExampleReport(1, "node od: boom")])


@pytest.mark.graph
def test_example_prints_a_failed_frame_on_stderr(capsys):
    """Minor #10: examples/run_graph.py prints a failed frame's line on
    stderr, as the CLI does, and the rest on stdout. No hardware: the
    example's Graph is replaced by one whose second frame fails."""
    import importlib.util
    import types
    dx_graph = load_dx_graph()
    path = PROJECT_ROOT / "src/bindings/python/dx_graph/examples/run_graph.py"
    spec = importlib.util.spec_from_file_location("run_graph_example", str(path))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.dx_graph = types.SimpleNamespace(Graph=ExampleGraph, GraphError=dx_graph.GraphError)
    assert example.main(["--graph", "unused.json"]) == 1
    out, err = capsys.readouterr()
    assert err == "frame 1: node od: boom\n"
    assert out == "frame 0: 1 node\n2 frames, 1 failed\n"


class ExampleTwoSourceGraph(ExampleGraph):
    streams = ("cam1", "cam2")

    def stream(self, source=None, limit=None):
        return iter([ExampleReport(0, "", "cam1"), ExampleReport(0, "node od: boom", "cam2")])


@pytest.mark.graph
def test_example_names_the_stream_of_a_two_source_graph(capsys):
    import importlib.util
    import types
    dx_graph = load_dx_graph()
    path = PROJECT_ROOT / "src/bindings/python/dx_graph/examples/run_graph.py"
    spec = importlib.util.spec_from_file_location("run_graph_example_two", str(path))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.dx_graph = types.SimpleNamespace(Graph=ExampleTwoSourceGraph,
                                             GraphError=dx_graph.GraphError)
    assert example.main(["--graph", "unused.json"]) == 1
    out, err = capsys.readouterr()
    assert err == "stream cam2 frame 0: node od: boom\n"
    assert out == "stream cam1 frame 0: 1 node\n2 frames, 1 failed\n"


class ExampleRefusingGraph(ExampleTwoSourceGraph):
    """stream() refuses what it is given, as a graph with several sources
    refuses source=."""

    def stream(self, source=None, limit=None):
        raise ValueError("source= replaces the one source of a single-source graph, and "
                         "this graph has 2 (cam1, cam2): pass sources={\"<id>\": uri, ...}")


@pytest.mark.graph
def test_example_prints_a_refused_stream_as_one_line(capsys):
    """A stream() the graph refuses (a ValueError, e.g. --input on a graph
    with several sources) is one line on stderr and exit 1, no traceback."""
    import importlib.util
    import types
    dx_graph = load_dx_graph()
    path = PROJECT_ROOT / "src/bindings/python/dx_graph/examples/run_graph.py"
    spec = importlib.util.spec_from_file_location("run_graph_example_refused", str(path))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.dx_graph = types.SimpleNamespace(Graph=ExampleRefusingGraph,
                                             GraphError=dx_graph.GraphError)
    assert example.main(["--graph", "unused.json", "--input", "clip.mp4"]) == 1
    out, err = capsys.readouterr()
    assert err == ("source= replaces the one source of a single-source graph, and this graph "
                   "has 2 (cam1, cam2): pass sources={\"<id>\": uri, ...}\n")
    assert out == ""


PORT_GRAPH = ('{"version":1,"nodes":[{"id":"cam","type":"source","uri":"sample/img/sample_street.jpg"},'
              '{"id":"drive","model":"yolopv2_384x640"},{"id":"sp","model":"superpoint_480x640"}],'
              '"edges":[{"from":"cam","to":"drive"},{"from":"cam","to":"sp"}]}')


@pytest.mark.graph
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_port_arrays_match_their_summaries(tmp_path, executor):
    import cv2
    import numpy
    for model in ("yolopv2_384x640", "superpoint_480x640"):
        if not (MODEL_DIR / _registry()[model]["dxnn_file"]).is_file():
            pytest.skip("{} not downloaded".format(model))
    dx_graph = load_dx_graph()
    path = tmp_path / "ports.json"
    path.write_text(PORT_GRAPH, encoding="utf-8")
    frame = cv2.imread(str(PROJECT_ROOT / "sample" / "img" / "sample_street.jpg"))
    with dx_graph.Graph(str(path), executor=executor) as g:
        d = g.run(frame).to_dict()
    drivable = d["nodes"]["drive"]["ports"]["drivable"]
    assert drivable["labels_array"].shape == (drivable["labels"]["rows"], drivable["labels"]["cols"])
    assert drivable["labels_array"].dtype == numpy.uint8
    assert numpy.count_nonzero(drivable["labels_array"]) == drivable["labels"]["nonzero"] > 0
    descriptors = d["nodes"]["sp"]["ports"]["descriptors"]["values_array"]
    assert descriptors.dtype == numpy.float32 and descriptors.shape[1] == 256
    assert descriptors.shape[0] == len(d["nodes"]["sp"]["payload"]["items"][0]["keypoints"])


FACE5_GRAPH = {
    "version": 1, "name": "face5-embeddings",
    "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_face_a1.jpg"},
              {"id": "face", "model": "scrfd500m"},
              {"id": "emb", "model": "arcface_mobilefacenet"}],
    "edges": [{"from": "cam", "to": "face"},
              {"from": "face", "to": "emb", "roi": {"align": "face5", "max": 1}}],
}

# (name, portrait): a reference, another photo of the same person, another
# person. sample_face_a1/_b are face_pair/1_reference/3_different byte for
# byte; face_pair/2_same is a byte copy of 1_reference, so the same-person
# photo is sample_face_a2 (a second, turned pose).
FACE5_PORTRAITS = (("reference", "sample/img/sample_face_a1.jpg"),
                   ("same", "sample/img/sample_face_a2.jpg"),
                   ("different", "sample/img/sample_face_b.jpg"))


def face5_input(tmp_path, name, portrait):
    """The portrait centred on a mid-grey canvas twice its size, as a PNG.
    scrfd500m finds no face in a close-up whose face fills the frame; with
    the margin it finds exactly one."""
    import cv2
    import numpy
    image = cv2.imread(str(PROJECT_ROOT / portrait), cv2.IMREAD_COLOR)
    assert image is not None, portrait
    rows, cols = image.shape[:2]
    canvas = numpy.full((2 * rows, 2 * cols, 3), 128, dtype=numpy.uint8)
    canvas[rows // 2:rows // 2 + rows, cols // 2:cols // 2 + cols] = image
    path = tmp_path / "{}.png".format(name)
    assert cv2.imwrite(str(path), canvas)
    return path


@pytest.mark.graph_e2e
def test_face5_aligned_embeddings_tell_the_same_face_from_another(tmp_path):
    """U-27: align "face5" on the NPU. The CLIs and dx_graph agree; each
    face is cut through a similarity transform (not an axis-aligned crop)
    onto a square as wide as its box, which the embedder resizes to 112x112;
    and the aligned embedding of the same person is closer than another
    person's."""
    import cv2
    import numpy
    graph = tmp_path / "face5.json"
    graph.write_text(json.dumps(FACE5_GRAPH), encoding="utf-8")
    require_artifacts(str(graph))
    dx_graph = load_dx_graph()
    embeddings = []
    for name, portrait in FACE5_PORTRAITS:
        source = face5_input(tmp_path, name, portrait)
        cli = {}
        for binary in ("sync", "async"):
            report = tmp_path / "{}_{}.json".format(name, binary)
            result = run("multi_model_graph_" + binary, "--graph", str(graph),
                         "--input", str(source), "--report", str(report))
            assert result.returncode == 0, result.stdout + result.stderr
            with open(str(report), encoding="utf-8") as handle:
                cli[binary] = json.load(handle)["frames"]
        assert cli["sync"] == cli["async"], name
        with dx_graph.Graph(str(graph), executor="async") as g:
            report = g.run(cv2.imread(str(source), cv2.IMREAD_COLOR))
        assert not report.error, report.error
        d = report.to_dict(arrays=False)
        assert d == cli["async"][0], name
        crops = d["roi_nodes"]["emb"]
        assert len(crops) == 1, (name, len(crops))
        origin = crops[0]["origin"]
        side = max(origin["src_box"][2], origin["src_box"][3])
        assert origin["crop_size"] == [side, side], origin
        assert abs(origin["inv_align"][1]) > 1e-4 or abs(origin["inv_align"][3]) > 1e-4, origin
        values = numpy.asarray(crops[0]["payload"]["values"], dtype=numpy.float64)
        assert values.shape == (512,) and numpy.all(numpy.isfinite(values)), values.shape
        print("\n  face5 {}: crop {}, inv_align {}".format(
            name, origin["crop_size"], [round(a, 3) for a in origin["inv_align"]]))
        embeddings.append(values / numpy.linalg.norm(values))
    same = float(embeddings[0] @ embeddings[1])
    other = float(embeddings[0] @ embeddings[2])
    print("  face5 cosine: same {:.3f}, different {:.3f}".format(same, other))
    assert same > other + 0.1, (same, other)


@pytest.mark.graph_e2e
@pytest.mark.parametrize("executor", ["sync", "async"])
def test_camera_streams_through_dx_graph(tmp_path, executor):
    """U-28: Graph.stream(source="camera:0") reads the USB camera inside
    C++, as the CLI's --input camera:0 does."""
    from test_graph_cli import CAMERA_OPEN_FAILED, camera_available
    graph = tmp_path / "camera.json"
    graph.write_text(json.dumps({
        "version": 1, "name": "camera-od",
        "nodes": [{"id": "cam", "type": "source", "uri": "sample/img/sample_people.jpg"},
                  {"id": "od", "model": "yolov8n"}],
        "edges": [{"from": "cam", "to": "od"}]}), encoding="utf-8")
    require_artifacts(str(graph))
    if not camera_available():
        pytest.skip("camera /dev/video0 is absent or busy (it does not open)")
    dx_graph = load_dx_graph()
    try:
        with dx_graph.Graph(str(graph), executor=executor) as g:
            reports = list(g.stream(source="camera:0", limit=5))
    except dx_graph.GraphError as error:
        if CAMERA_OPEN_FAILED in str(error):  # taken since camera_available()
            pytest.skip("camera /dev/video0 became busy before the stream opened it")
        raise
    assert [r.frame_index for r in reports] == list(range(5))
    first = reports[0].frame.shape
    for report in reports:
        assert not report.error, report.error
        # Any camera: a non-empty 3-channel frame, the same size every frame.
        assert report.frame.ndim == 3 and report.frame.shape[2] == 3 and report.frame.size > 0
        assert report.frame.shape == first, (report.frame.shape, first)
