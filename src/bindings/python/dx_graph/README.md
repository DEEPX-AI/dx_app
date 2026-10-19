# `dx_graph` — the multi-model graph from Python

`dx_graph` runs a multi-model graph JSON (see
[`src/cpp_example/multi_model_graph/README.md`](../../../cpp_example/multi_model_graph/README.md))
from Python. It is not a second implementation: the module links the same
engine, the same executors, the same report serializer and the same renderer
as `multi_model_graph_{sync,async}`, so a Python run *is* a CLI run — the
same reports, the same pictures, the same error messages.

```python
import dx_graph

with dx_graph.Graph("src/cpp_example/multi_model_graph/fanout_od_seg_depth.json") as g:
    for report in g.stream():                 # the graph's own source, pipelined
        print(report.frame_index, report.error or "ok")
        canvas = g.render(report)             # numpy BGR, what --output writes
```

Run only: Python loads a graph, feeds it frames (or lets it read its own
source) and gets results. Graphs are written in JSON, as for the CLI; there is
no API to build or edit one, and no Python stages.

---

## Build

The module is built by the dx_app build, **for one Python interpreter**:
`DXAPP_PYTHON`. It needs `pybind11` (3.x; built and tested with 3.1.0) and
the interpreter's development headers *for that interpreter*, and `numpy` at
run time.

```bash
# 1) pybind11 and numpy for the interpreter that will import dx_graph
/path/to/python -m pip install pybind11 numpy

# 2) build everything for THAT interpreter and ship it into bin/
./build.sh --all --python_exec /path/to/python

# 3) use it
PYTHONPATH=bin/python /path/to/python -c "import dx_graph; print(dx_graph.build_info())"
```

What `./build.sh` does with the module:

- **`./build.sh --all`** configures `build_x86_64`, builds every target (the
  module and the `multi_model_graph_{sync,async}` CLI included), installs
  them into `build_x86_64/release`, and copies that into the repository: the
  CLI binaries land in `bin/`, the package in **`bin/python/dx_graph`**
  (`_dx_graph.<ext>` and `__init__.py`) and the example beside it in
  `bin/python/dx_graph_examples/run_graph.py`. It then pip-installs
  `dx_postprocess` into the `--python_exec` interpreter, as it always has.
- **`./build.sh` with no mode** is `--minimal`: only the `run_demo.sh`
  C++ targets are built and copied to `bin/`. It neither builds nor ships
  `dx_graph` (nor the multi_model_graph CLI). `--target` and `--category`
  copy executables only, so they do not ship the package either: use `--all`.
- **The interpreter**: `--python_exec <python>` sets it (passed on as
  `-DDXAPP_PYTHON`), else `--venv_path <venv>` (its `bin/python`), else the
  active virtualenv's `python` (`$VIRTUAL_ENV`). With none of these,
  `build.sh` passes nothing and CMake keeps its own choice: the value cached
  in `build_x86_64`, or on a first configure whatever `python3` is first on
  `PATH` — on a machine with conda that is conda's Python, and the module it
  builds does not import in any other interpreter (the
  `_dx_graph.cpython-3XX-...so` file name carries the version it was built
  for). So always name it on the first build.
- **`./build.sh --clean`** pip-uninstalls `dx_postprocess` from the
  `--python_exec` interpreter, deletes `build_x86_64` — the cached
  `DXAPP_PYTHON` with it — together with `bin/`, `lib/` and `include/` (so
  `bin/python/dx_graph` too), then builds as above. A cross build
  (`--arch` other than the host's) deletes only its own build directory. Without
  `--clean`, `bin/python/dx_graph` is copied over, not emptied: a module
  built earlier for another interpreter stays beside the new one until
  `--clean` or `rm -rf bin/python/dx_graph`.

A cross build (`--arch aarch64`) skips the module (`dx_graph Python module
skipped: cross build ...`) unless `-DDXAPP_CROSS_PYTHON_GRAPH=ON` and the
target Python's `-DPython_INCLUDE_DIR` are given; that path is not covered
by the release tests.

The configure step names what it chose:

```
-- dx_graph Python module: .../build_x86_64/python/dx_graph (Python 3.12.3, pybind11 3.1.0)
```

or `-- dx_graph Python module skipped: no pybind11 for ...` /
`... skipped: no Python development headers (Development.Module) for ...`
— the CLI builds either way. `-DDXAPP_BUILD_PYTHON_GRAPH=OFF` turns the
module off. `dx_graph.build_info()` reports the Python version the module was
built for and the repository it belongs to; importing it in another
interpreter raises an `ImportError` naming that interpreter's `sys.version`
and the `_dx_graph*.so` files the package holds.

Building by hand instead of `build.sh`: the toolchain file is required, and
the result stays in the build tree until you copy it:

```bash
cmake -S . -B build_x86_64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.x86_64.cmake \
      -DDXAPP_PYTHON=/path/to/python
ninja -C build_x86_64
rm -rf bin/python/dx_graph bin/python/dx_graph_examples && mkdir -p bin/python
cp -r build_x86_64/python/dx_graph build_x86_64/python/dx_graph_examples bin/python/
```

The module is built for the interpreter given at build time
(`-DDXAPP_PYTHON`, or `./build.sh --python_exec`); any interpreter of the same
Python version imports it.

---

## API

### `dx_graph.Graph(path, model_dir=None, executor="async", max_frames_in_flight=16, max_jobs_per_stage=0, stall_timeout_ms=0)`

Parses, validates and builds the graph — every model is loaded here. Raises
`dx_graph.GraphError` exactly where the CLI would exit 1 before running.

| Argument | CLI equivalent | Notes |
|---|---|---|
| `path` | `--graph` | the graph JSON |
| `model_dir` | `--model-dir` | default `<repo>/assets/models`, as the CLI |
| `executor` | the binary | `"async"` (`multi_model_graph_async`) or `"sync"` |
| `max_frames_in_flight` | `--max-inflight` | **async only**; at least 1 |
| `max_jobs_per_stage` | `--max-jobs-per-stage` | **async only**; jobs one stage may have outstanding; 0 = no limit |
| `stall_timeout_ms` | `--stall-timeout-ms` | **async only**; raise `RuntimeError` when work is outstanding and nothing completes for this long; 0 = wait forever (see *Ctrl-C* below) |

The three async-only options take the same values as the CLI's
`--max-inflight`, `--max-jobs-per-stage` and `--stall-timeout-ms`. They are
validated for a sync Graph too, and then unused (where
`multi_model_graph_sync` refuses them with exit 2). `Graph` is a context
manager; `close()` releases the executor and every model and is
idempotent. `g.executor` is `"sync"` or `"async"`; `g.options` (a read-only
mapping) holds the executor options in use: the three above for an async
Graph, **empty for a sync Graph**, whose executor has none.

`g.streams` is a tuple of the graph's source node ids, in declaration
order: one stream each, as in the CLI (see *Several sources: one stream
each* in the multi_model_graph README). A one-source graph has one, such as
`("cam",)`.

**Nothing carries over from one call to the next.** Every `run()` and every
`stream()` starts with every node's tracker reset (no tracks, ids from 0),
so a reused Graph gives exactly what a fresh one gives. Track ids follow
objects across the frames of *one* stream; to track across frames, pass them
to one `stream(frames=...)`. With several sources a tracked node has one
tracker per stream, so ids never cross from one source's stream to
another's.

### `g.run(frame, stream=None) -> Report`

One frame, start to finish, numbered 0 as the CLI numbers a single image —
an independent image, as in the CLI's single-image run: trackers are reset
first, so no track carries over from an earlier `run()` or `stream()`.
`frame` is a numpy `uint8` array of shape `(H, W, 3)` in BGR order (what
`cv2.imread` returns); any strides are accepted, the frame is copied.
Anything else raises `TypeError` (not an array, not `uint8`) or `ValueError`
(wrong shape).

`stream` names the source the frame belongs to, one of `g.streams`; the
frame then runs the nodes of that source's stream. On a one-source graph it
may be left out. With several sources it is required, and leaving it out is
a `ValueError`: `run() on a graph with 2 sources (cam1, cam2) needs
stream=: the id of the source this frame belongs to`. An id that is not a
source is a `ValueError` too: `'camX' is not a source of this graph; its
sources: cam1, cam2`.

### `g.stream(frames=None, *, source=None, sources=None, limit=None) -> iterator of Report`

Reports in frame order, pipelined: under the async executor up to
`max_frames_in_flight` frames are in flight, so the report handed out is not
necessarily for the frame just read.

- `frames=None` — the graph reads its own source node's `uri`, frame by frame,
  inside C++ (no numpy round trip). `source=` overrides the `uri` exactly as
  the CLI's `--input` does: an image, a video, `camera:<N>`, `rtsp://...`.
- `frames=<iterable>` — any iterable of numpy frames, as for `run()`.
- `limit` — the CLI's `--frames`: stop after that many frames; `None` or 0 =
  all. It is checked after each submitted frame, so the item after the last
  one is never taken from `frames`.

Each stream is numbered from 0 and starts on fresh trackers, as the CLI starts
on its input: streaming the same frames twice through one Graph gives the same
reports twice.

A graph with several sources runs one stream per source, as the CLI does:

- `sources={"cam1": uri, ...}` — the graph reads its sources through the
  CLI's own reader, in turn: one frame from each stream still reading, in
  declaration order, as the CLI's `--input cam1=<uri>`. A source left out
  reads its own `uri`, and `limit` is per stream, as the CLI's `--frames`.
  Reports come out in that order, each stream numbered from 0 on its own. On
  a one-source graph `sources={"cam": uri}` is `source=uri`.
- no `frames`, `source` or `sources` — the graph reads its own sources, as
  `sources={}`.
- `frames=` — an iterable of `(source_id, frame)` pairs: each frame is
  submitted to its source's stream, the streams numbered on their own, and
  the reports come out in the order of the pairs. An item that is not a pair
  is a `TypeError`, an id that is not a source a `ValueError`. `limit` is
  refused with pairs (`ValueError`): "N frames per stream" could not be
  honoured without taking items past the N-th from your iterator.
- `source=` is refused with a `ValueError`, since it does not say which
  source it replaces: `source= replaces the one source of a single-source
  graph, and this graph has 2 (cam1, cam2): pass sources={"<id>": uri, ...}`.

`frames` with `sources`, and `source` with `sources`, are `ValueError`s, as
`frames` with `source` is. A source that gives no frame at all raises
`dx_graph.GraphError`, with the CLI's message (`source "<uri>": produced no
frames`). With several sources it does not stop the others, as in the CLI:
every report of the other streams comes out first, and once all have ended
one `GraphError` (`GRAPH_SCHEMA`) names every empty source, the CLI's lines
joined by newlines in declaration order. A stream you leave early (`break`,
`close()`) raises nothing for it.

Leaving a stream early — `break`, an exception in the loop body, closing the
iterator — finishes the frames in flight and discards their reports, so the
Graph is usable again at once. That happens when the *iterator* is closed. A
`for` loop over `g.stream(...)` closes it when the loop is left; an iterator
you keep a name for stays open, and keeps the Graph locked, until you close
it or it is collected:

```python
it = g.stream()
for r in it:
    break          # `it` still holds the Graph ...
it.close()         # ... until here
```

`g.close()` while a stream on the same thread is suspended ends that stream at
its next step: nothing more is read or submitted, the frames in flight are
finished and dropped, and the models are released.

### `g.render(report) -> numpy.ndarray`

The CLI's `--output` image for `report`: every result drawn onto **the
report's own frame** — not the frame most recently read, which under
pipelining belongs to a later report. `uint8`, `(H, W, 3)`, BGR, so
`cv2.imwrite(path, g.render(r))` writes what the CLI writes. `render()` may
be called inside the same Graph's `stream()` loop. It needs the Graph open:
after `close()` (or the end of its `with` block) it raises
`RuntimeError("dx_graph.Graph is closed")`, even for a report taken earlier —
render what you need before closing.

`render()` draws ports as the CLI does: the `yolopv2_384x640` masks in green
(drivable) and red (lane), a 3DDFA `pose` as `vec[3]` and handedness as
`Left: 97%` at the crop. SuperPoint's descriptors are not drawn.

Node `"params"` may hold strings and arrays of strings for the keys a model
reads as text (today `class_names`), and ESPCN in a graph is tiled as its
runner tiles it, so it hands off the input size × the factor. Both are
described in the
[multi_model_graph README](../../../cpp_example/multi_model_graph/README.md)
(*Parameter precedence*, *Coordinates after a hand-off*).

### `Report`

| Member | |
|---|---|
| `frame_index` | the frame's index in its stream (0, 1, ...) |
| `stream` | the id of the source node this frame came from |
| `error` | `""` when every stage succeeded, else the first failing stage's message |
| `frame` | a numpy copy of the frame this report was computed on |
| `to_dict(arrays=True)` | this frame's entry of the CLI's `--report` JSON, as a dict |

With `arrays=True`, every dense payload summary in the dict — `labels`,
`values`, `image`, a box's `mask` — gains a sibling `<key>_array` holding a
numpy copy of the data it summarizes (`d["nodes"]["seg"]["payload"]["labels_array"]`,
...). `to_dict(arrays=False)` is exactly the CLI entry: it carries `"stream"`
exactly when the CLI's report does, on a graph with several sources.

A result with **output ports** (`yolopv2_384x640`, `superpoint_480x640`, the two 3DDFA
models, `mediapipe-hands-lite_224x224`, the two `ppmatting` models and the five
gallery models; see *Output ports* in the multi_model_graph README) has a `"ports"` entry next to its `"payload"`, as in the CLI's
report, and each dense port payload gets its `<key>_array` too:

- `d["nodes"]["drive"]["ports"]["lane"]["labels_array"]` is the lane mask:
  `uint8` 0/1, at the size of the image `yolopv2_384x640` ran on, which is the
  source size when it reads the source (`(576, 768)` for a 768×576 image).
  `"drivable"` is the drivable-area mask, likewise.
- `d["nodes"]["sp"]["ports"]["descriptors"]["values_array"]` is `float32`
  N×256, row i = the descriptor of keypoint i (`(500, 256)` at SuperPoint's
  default `top_k`). A frame with no keypoints gives an empty `(0, 0)`
  array of dtype `uint8`, not `float32`.
- `pose` (3DDFA) and `handedness` (hand landmark) are plain JSON: a list of
  floats and a list of items with `class_name`, `class_id` and `confidence`.
  In a cascade they run per crop, so their ports are on each ROI result:
  `d["roi_nodes"]["lmk"][0]["ports"]["handedness"]`.

A result without ports has no `"ports"` key.

A graph with logic nodes (see *Logic nodes* in the multi_model_graph README)
gives a frame whose nodes raised events an `"events"` list in `to_dict()`,
exactly as in the CLI's report. `dx_graph` counts their time at 30 frames
per second of each stream (it does not know the source's rate). Printing
`[EVENT]` lines, `--events`, `--loop` and the event panel are CLI options.

A report you keep after its stream has moved on (`reports =
list(g.stream(...))`) keeps its own frame: its `frame`, `to_dict()` and
`render()` give what they gave inside the loop, under both executors.

A NaN, +Inf or -Inf value in a report is the string `"NaN"`, `"Infinity"`
or `"-Infinity"` in the dict, exactly as in the CLI's `--report` file
(strict JSON has no such numbers); `float()` turns it back into the number.
`to_dict()` does not convert them. Two such dicts compare equal with `==`
(the strings are equal), while the C++ `FrameReport::operator==` treats NaN
as never equal.

### `dx_graph.GraphError`

A `ValueError` subclass for a graph that cannot be built or a source that
cannot be read. `.code` is the CLI's error code (`"GRAPH_EDGE"`,
`"MODEL_MISSING"`, `"GRAPH_SCHEMA"`, ...; the table is in the
multi_model_graph README), `str()` the message the CLI prints for the same
file. `.code` is `"MODEL_LOAD"` when the runtime cannot load a model whose
`.dxnn` is present, for example because the device memory is full (see
*Device memory*), or because the file is a `.dxnn` container v9 and this
DX-RT is older than 3.5.0 (the message is the CLI's: `... .dxnn container v9
needs DX-RT >= 3.5.0, but this runtime is 3.4.1. Use the v8 file (dxnn/2_4_0)
or upgrade DX-RT.`). Other engine failures,
such as a `stall_timeout_ms` stall, are `RuntimeError`.

---

## Parity guarantee

`tests/cpp_example/test_graph_python.py` checks on real hardware that:

- for **every sample graph** and both executors, `run()` on the graph's image
  gives `to_dict(arrays=False)` **equal** to the CLI's `--report` entry;
- `stream()` over a moving 8-frame video equals the CLI's `--report` for the
  same video, frame by frame, for both executors, on a tracking cascade
  (`cascade_od_reid_track.json`) and an image hand-off
  (`handoff_denoise_od.json`); a stream of numpy frames gives the same reports
  under both executors;
- a **reused Graph** equals a fresh one: two streams of the same video through
  one Graph on the tracking cascade both equal the CLI's `--report`, and a
  `run()` after a stream equals a fresh Graph's `run()`, both executors;
- on the two-source sample (`multistream_od_two_sources.json`),
  `stream(sources=...)` over two videos equals the CLI's `--report` for the
  same `--input cam1=... --input cam2=...`, order included, for both
  executors, and a stream of `(source_id, frame)` pairs gives the same
  reports under both executors;
- `render()` equals the CLI's `--output`, **pixel for pixel**, for every frame
  of a pipelined stream and for a single image;
- `GraphError` codes and messages equal the CLI's stderr, word for word.

The report schema is not re-implemented in Python: `to_dict()` parses the
JSON that the CLI's own serializer writes.

---

## Threading and the GIL

- Every engine call — building the graph (model loading), submitting, waiting,
  reading the source, rendering — runs **with the GIL released**, so other
  Python threads keep running while the NPU works.
- One `Graph` is used by **one thread at a time**. A call from a second thread
  while one is running raises `RuntimeError("dx_graph.Graph is already running
  in another thread")`; a `run()` or a second `stream()` inside a stream loop
  on the same Graph raises `RuntimeError("dx_graph.Graph is already in use by
  this thread ...")`. `render()` is the exception: a stream loop may call it.
  Use one `Graph` per thread for parallel work.
- `run()` refuses while a stream's frames or reports are still pending: it
  never returns a report that belongs to another frame.

## Ctrl-C and `stall_timeout_ms`

Under the async executor, a Ctrl-C (or any Python signal handler that raises)
arriving while the engine waits is noticed within 50 ms (while nothing
completes, the Graph takes the GIL to check for signals at most once per
50 ms, so a waiting Graph does not compete with your other threads); the call
finishes the frames it has in flight — dxrt cannot cancel a job — and then
raises the exception (`KeyboardInterrupt`), leaving the Graph usable. If no
stage has completed anything for a second after the signal, the stages still
holding work are printed to stderr exactly as the CLI prints them
(`interrupted: still waiting on stage(s): ...`). Under the sync executor the
signal is raised when the current frame returns.

A runtime callback that never arrives would block an async wait forever. Pass
`stall_timeout_ms` to bound it: the call then raises `RuntimeError`
(`AsyncExecutor stalled: no completion for N ms; waiting on ...`), the reports
still pending are dropped, and the Graph replaces its executor, so it stays
usable. If a signal was caught during that wait, the signal's exception is
raised instead of the `RuntimeError`.

SIGTERM is where the CLI and a Python program differ. The CLI handles
SIGTERM like Ctrl-C: it finishes the frames in flight and finalizes
`--report` and `--output`. `dx_graph` installs no signal handler, so inside
a Python program Python's default SIGTERM action still applies and the
process ends at once. To have SIGTERM behave as Ctrl-C does above, install
a handler that raises, with `signal.signal(signal.SIGTERM, ...)`.

## Device memory

Every model node loads its own engine on the NPU, even two nodes that name
the same model, and other processes on the NPU draw on the same device
memory. A graph whose engines do not fit raises `GraphError` with `.code ==
"MODEL_LOAD"` from the constructor, naming the first node that did not fit:
on DX-M1, a second `realesrgan-x2_192x192` next to `yolov8-n_640x640`, `resnet50_224x224` and a first
`realesrgan-x2_192x192` cannot load. Nothing checks this in advance (the CLI's
`--check` cannot either: DX-RT has no API for a model's memory before it
loads). A node's `"engine"` key lowers what a graph needs: `"share": true`
on nodes with the same model gives them one engine, and `"buffers": N`
shrinks a model's buffer pool (DX-RT's default is 6); otherwise remove a
node or use a smaller model. See the multi_model_graph README's *Device
memory*.

---

## Example

[`examples/run_graph.py`](examples/run_graph.py) is the Python counterpart of
the CLI: it streams a graph, prints one line per frame, writes each frame's
rendering (PNG through OpenCV when `cv2` is installed, else a PPM written
with numpy alone) and ends with the CLI's `N frames, F failed` line. As in
the CLI, a failed frame's `frame N: <error>` line goes to stderr, the rest to
stdout. A graph with several sources is read through all of them, and each
line and file name names the stream: `stream cam1 frame 0: 3 nodes`,
`frame_cam1_000000.png`. `--input` replaces the only source of a one-source
graph; on a graph with several, the example prints the `source=` refusal on
one line and exits 1.

```bash
PYTHONPATH=bin/python python3 src/bindings/python/dx_graph/examples/run_graph.py \
    --graph src/cpp_example/multi_model_graph/cascade_od_reid_track.json \
    --input my_video.mp4 --frames 100 --output-dir /tmp/renders
```

The build installs the same script as
`bin/python/dx_graph_examples/run_graph.py` (`PYTHONPATH=bin/python python3
bin/python/dx_graph_examples/run_graph.py ...`).

Options: `--graph`, `--model-dir`, `--input`, `--max-inflight` and `--frames`
work as the CLI options of the same name. The CLI has neither of the other
two: `--executor sync|async` picks the executor, which on the CLI is the
choice of binary (`multi_model_graph_sync` or `multi_model_graph_async`),
and `--output-dir` writes each frame's rendering into a directory, where
the CLI's `--output <file>` writes one video, or one image per frame named
after the file.

## Performance

Pipelining survives the binding: an `executor="async"` Graph runs the CLI's
own asynchronous executor (`consumer::AsyncFrameExecutor`, the one
`multi_model_graph_async` uses), with up to `max_frames_in_flight` frames in
flight. No throughput figure is published for the binding; measure it on
your own graph and hardware.

## Limitations

- One thread per `Graph` at a time (see above).
- Not pip-installable yet: the module is built by `./build.sh --all` (or the
  CMake build) and used from `bin/python` through `PYTHONPATH`. No wheels, no
  Windows build, and a cross build skips it unless the target's Python is
  named.
- Run only — no graph construction or editing API, no Python stages.
- No advance check that a graph's models fit in device memory (see *Device
  memory*), and no SIGTERM handler (see *Ctrl-C and `stall_timeout_ms`*).
- With several sources: no `source=` (name each source in `sources=`), no
  `limit` with `(source_id, frame)` pairs, and one `max_frames_in_flight`
  window shared by every stream, so a slow stream holds back the others (see
  the multi_model_graph README's *Several sources: one stream each*).
