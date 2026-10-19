"""dx_graph - run a dx_app multi-model graph JSON on the DEEPX NPU from Python.

Build: `./build.sh --all --python_exec <python>` builds it for that Python
and ships it as bin/python/dx_graph; add bin/python to PYTHONPATH.

    import dx_graph

    with dx_graph.Graph("src/cpp_example/multi_model_graph/fanout_od_seg_depth.json") as g:
        report = g.run(frame)            # numpy uint8, shape (H, W, 3), BGR
        d = report.to_dict()             # the CLI --report entry, plus *_array numpy fields
        for report in g.stream():        # pipelined, from the graph's own source
            print(report.frame_index, report.error)
            canvas = g.render(report)    # the CLI's --output image, numpy BGR
        for report in g.stream(sources={"cam1": a, "cam2": b}): ...  # one stream per source

Every message, error code and report field is the multi_model_graph CLI's own:
the module runs the CLI's code, not a copy of it.
"""
import collections.abc
import json
import operator
import threading
import types


def _load_core():
    """Import the C++ core, or say which interpreter failed to load which
    file - not Python's "partially initialized module (most likely due to a
    circular import)", which is what a missing _dx_graph looks like."""
    try:
        from . import _dx_graph as core
    except ImportError as error:
        import glob
        import importlib.machinery
        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        built = sorted(os.path.basename(p)
                       for p in glob.glob(os.path.join(here, "_dx_graph*.so")))
        ours = ["_dx_graph" + s for s in importlib.machinery.EXTENSION_SUFFIXES]
        loadable = [name for name in built if name in ours]
        head = ("dx_graph: cannot load its C++ core _dx_graph in this Python "
                "({}, {}). Found in {}: {}. ".format(
                    sys.version, sys.executable, here,
                    ", ".join(built) if built else "no _dx_graph*.so"))
        if loadable:
            raise ImportError(head + "Loading {} failed: {}".format(
                loadable[0], error)) from error
        if not built:
            raise ImportError(
                head + "The package has no compiled core: build it for this "
                "Python (./build.sh --all --python_exec <python>).") from None
        raise ImportError(
            head + "This Python loads only {}: the module was built for "
            "another interpreter. Rebuild it for this one (./build.sh "
            "--python_exec <python>, or -DDXAPP_PYTHON=<python>), or run the "
            "interpreter it was built for.".format(" or ".join(ours))) from None
    return core


_dx_graph = _load_core()

__version__ = "0.1.0"

__all__ = ["Graph", "GraphError", "Report", "build_info", "__version__"]

#: A graph that cannot be built (``ValueError`` subclass). ``.code`` is the
#: CLI's error code (``"GRAPH_EDGE"``, ``"MODEL_MISSING"``, ...); ``str()`` is
#: the message the CLI prints for the same file.
GraphError = _dx_graph.GraphError

_EXECUTORS = ("sync", "async")


def build_info():
    """The Python version this module was built for and the repository root."""
    return dict(_dx_graph.build_info())


class Report(object):
    """One frame's result, and the frame it belongs to.

    ``frame_index`` and ``error`` are the report's own; ``stream`` names the
    source it came from; ``frame`` is a numpy copy of the frame the report
    was computed on (``None`` if there is none).
    """

    __slots__ = ("_r",)

    def __init__(self, core):
        self._r = core

    @property
    def frame_index(self):
        return self._r.frame_index

    @property
    def error(self):
        """"" when every stage succeeded, else the first failing stage's message."""
        return self._r.error

    @property
    def stream(self):
        """The id of the source node this frame came from."""
        return self._r.stream

    @property
    def frame(self):
        """The report's own frame as a new numpy BGR array, or None."""
        return self._r.frame()

    def to_dict(self, arrays=True):
        """This frame's entry of the CLI's ``--report`` JSON, as a dict.

        With ``arrays=True`` every dense payload summary (``labels``,
        ``values``, ``image``, ``mask``) gains a sibling ``<key>_array``
        holding a numpy copy of the data it summarizes. That includes the
        payloads of a result's output ports (``"ports"`` -> name), e.g.
        ``d["nodes"]["drive"]["ports"]["drivable"]["labels_array"]``.
        """
        d = json.loads(self._r.json())
        if arrays:
            for path, array in self._r.arrays():
                container = d
                for key in path[:-1]:
                    container = container[key]
                container[path[-1] + "_array"] = array
        return d

    def __repr__(self):
        return "<dx_graph.Report frame_index={} error={!r}>".format(
            self.frame_index, self.error)


class Graph(object):
    """A graph JSON, parsed, validated and built - every model loaded.

    Raises GraphError exactly where the multi_model_graph CLI would exit 1
    before running, with the same code and message. One Graph is used by one
    thread at a time: a concurrent call raises RuntimeError, and so does a
    call from inside a stream() loop on the same Graph - except render(),
    which a stream loop may call.

    ``max_frames_in_flight``, ``max_jobs_per_stage`` and ``stall_timeout_ms``
    are options of the async executor only: a sync Graph validates them and
    does not use them, and its ``options`` is empty.

    Nothing carries over between calls: every run() and every stream()
    starts with every node's tracker reset, so a reused Graph gives exactly
    what a fresh one gives. Track ids follow objects across the frames of
    ONE stream, never from one call to the next.
    """

    def __init__(self, path, model_dir=None, executor="async",
                 max_frames_in_flight=16, max_jobs_per_stage=0,
                 stall_timeout_ms=0):
        if executor not in _EXECUTORS:
            raise ValueError("executor must be \"sync\" or \"async\", got {!r}"
                             .format(executor))
        max_frames_in_flight = _require_int("max_frames_in_flight", max_frames_in_flight, 1)
        max_jobs_per_stage = _require_int("max_jobs_per_stage", max_jobs_per_stage, 0)
        stall_timeout_ms = _require_int("stall_timeout_ms", stall_timeout_ms, 0)
        self._lock = threading.Lock()
        self._owner = None  # thread ident of the lock holder, for the message
        self._core = None
        self._core = _dx_graph._Graph(
            str(path), None if model_dir is None else str(model_dir), executor,
            max_frames_in_flight, max_jobs_per_stage, stall_timeout_ms)
        self._executor = self._core.executor
        self._options = types.MappingProxyType(dict(self._core.options))
        self._streams = tuple(self._core.stream_ids)

    @property
    def executor(self):
        """"sync" or "async"."""
        return self._executor

    @property
    def options(self):
        """The executor options in use (read-only): max_frames_in_flight,
        max_jobs_per_stage and stall_timeout_ms for an async Graph; empty for
        a sync Graph, whose executor has none."""
        return self._options

    @property
    def streams(self):
        """The source node ids, in declaration order: one stream each."""
        return self._streams

    def run(self, frame, stream=None):
        """One frame, start to finish -> Report (frame_index 0, as the CLI
        numbers a single image). ``frame``: numpy uint8, shape (H, W, 3).

        With several sources, ``stream`` names the source the frame belongs
        to (one of ``streams``) and is required; with one, it may be left out.

        An independent image, as the CLI's single-image run: every tracker
        is reset first, so no track carries over from an earlier run() or
        stream(). To track across frames, pass them to stream(frames=...)."""
        index = self._stream_index(stream, "run()")
        core = self._enter()
        try:
            return Report(core.run_frame(frame, index))
        finally:
            self._leave()

    def stream(self, frames=None, *, source=None, sources=None, limit=None):
        """Reports in frame order, pipelined -> iterator of Report.

        ``frames=None`` reads the graph's own source node ``uri`` (or
        ``source``, which overrides it exactly as the CLI's ``--input``
        does), frame by frame inside C++. Otherwise ``frames`` is any
        iterable of numpy frames, as for ``run()``. ``limit`` is the CLI's
        ``--frames``: stop after that many frames; ``None`` or 0 = all.

        A graph with several sources runs one stream per source, as the CLI
        does. With ``frames=None`` it reads every source in turn - one frame
        from each stream still reading, in declaration order - through the
        CLI's own reader; ``sources={"cam1": uri, ...}`` overrides some of
        their ``uri``s, as the CLI's ``--input cam1=<uri>`` (a source left
        out reads its own), and ``limit`` is then per stream, as the CLI's
        ``--frames``. ``source=`` is refused on such a graph: it would not
        say which source it replaces. On a one-source graph
        ``sources={"cam": uri}`` is ``source=uri``. ``frames`` on a graph
        with several sources is an iterable of ``(source_id, frame)`` pairs,
        each frame submitted to its source's stream; ``limit`` is refused
        with it, since "N frames per stream" could not be honoured without
        pulling past the caller's N-th item. A source that gives no frame
        is the CLI's "produced no frames" GraphError: with several sources
        the other streams' reports all come out first, then one GraphError
        names every empty source (the CLI's lines, joined by newlines) -
        unless the stream was left early.

        Each stream is numbered from 0 and starts with every tracker reset,
        as the CLI starts on its input: track ids follow objects across this
        stream's frames, and a second stream on the same Graph gives what
        the first gave for the same frames. With several sources each
        source's stream is numbered, and tracked, on its own.

        The Graph is held for the iterator's lifetime. Leaving early - break,
        an exception, closing the iterator - finishes the frames in flight
        and discards their reports, so the Graph is reusable at once. That
        happens when the iterator itself is closed: one that is still
        referenced stays open, and keeps the Graph locked, until
        ``it.close()`` or until it is collected::

            it = g.stream()
            for r in it:
                break        # `it` still holds the Graph ...
            it.close()       # ... until here

        A ``for`` loop over ``g.stream(...)`` itself has no such name and
        releases the Graph as soon as the loop is left. The ``limit`` is
        checked after each submitted frame, as the CLI checks ``--frames``:
        the item after the last one is never taken from ``frames``.
        ``close()`` while this thread's stream is suspended ends the stream
        at its next step: nothing more is read or submitted, the frames in
        flight are finished and dropped, and the models are released.
        ``render()`` may be called from inside the loop.
        """
        if frames is not None and source is not None:
            raise ValueError("pass frames or source, not both")
        if frames is not None and sources is not None:
            raise ValueError("pass frames or sources, not both")
        if source is not None and sources is not None:
            raise ValueError("pass source or sources, not both")
        limit = 0 if limit is None else _require_int("limit", limit, 0)
        several = len(self._streams) > 1
        if source is not None:
            if several:
                raise ValueError(
                    "source= replaces the one source of a single-source graph, and "
                    "this graph has {} ({}): pass sources={{\"<id>\": uri, ...}}".format(
                        len(self._streams), ", ".join(self._streams)))
            source = str(source)
        if sources is not None:
            sources = self._check_sources(sources)
            if not several:  # one source: exactly source=
                source, sources = sources.get(self._streams[0]), None
        if frames is not None:
            if several and limit:
                raise ValueError(
                    "limit counts each stream's frames when the graph reads its "
                    "sources; with frames= on a graph with several sources, pass "
                    "only the frames you want")
            frames = iter(frames)
        elif several and sources is None:
            sources = {}  # the graph's own sources
        return self._stream(frames, source, sources, limit)

    def _stream(self, frames, source, sources, limit):
        # A generator: nothing below runs, and no lock is taken, until the
        # first next(). An iterator that is never started holds nothing.
        #
        # After every yield the caller may have called close() (R3(b)): the
        # generator then returns at once - nothing more is read or submitted
        # - and the finally clause abandons what is in flight and releases
        # the models, which close() could not do while this held the lock.
        drained = False
        core = self._enter()
        try:  # at once: no statement between taking the lock and the try
            if sources is not None:
                src = core.open_sources(sources, limit)
            elif frames is None:
                src = core.open_source(source)
            else:
                src = None
            pairs = len(self._streams) > 1  # frames are (source_id, frame)
            count = 0
            while True:
                if src is not None:
                    if not src.submit_next(core):
                        break
                else:
                    frame = next(frames, _END)
                    if frame is _END:
                        break
                    if pairs:
                        stream, frame = self._pair(frame)
                        core.submit(frame, stream)
                    else:
                        core.submit(frame)
                count += 1
                for report in _drain_ready(core):
                    yield report
                    if self._core is not core:
                        return
                # After the submit, as the CLI checks --frames: the item
                # after the last one is never pulled from `frames` (R3(a)).
                # With `sources` the reader applies the limit, per stream.
                if limit and sources is None and count >= limit:
                    break
            core.finish()
            for report in _drain_ready(core):
                yield report
                if self._core is not core:
                    return
            # Only now is nothing left: a break during the tail above still
            # leaves reports behind, and abandon() must take them.
            drained = True
            if sources is not None:
                # Spec R9, as the CLI: a source that gave no frame did not
                # stop the others; it is named now that they have all ended.
                src.raise_if_empty()
        finally:
            try:
                if not drained:
                    core.abandon()
            finally:
                try:
                    if self._core is not core:  # closed while we held it
                        core.close()
                finally:
                    self._leave()

    def _stream_index(self, stream, what):
        """The streams() index of source id `stream`; None names the only one."""
        if stream is None:
            if len(self._streams) > 1:
                raise ValueError(
                    "{} on a graph with {} sources ({}) needs stream=: the id of the "
                    "source this frame belongs to".format(
                        what, len(self._streams), ", ".join(self._streams)))
            return 0
        if stream not in self._streams:
            raise ValueError("{!r} is not a source of this graph; its sources: {}".format(
                stream, ", ".join(self._streams)))
        return self._streams.index(stream)

    def _pair(self, item):
        """A (source_id, frame) item of frames= -> (stream index, frame)."""
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
            raise TypeError(
                "a graph with several sources takes frames as (source_id, frame) "
                "pairs, got {}".format(type(item).__name__))
        return self._stream_index(item[0], "frames="), item[1]

    def _check_sources(self, sources):
        """sources= -> {source id: str(uri)}, every key a source of this graph."""
        if not isinstance(sources, collections.abc.Mapping):
            raise TypeError("sources must be a mapping of source id to uri, got {}".format(
                type(sources).__name__))
        checked = {}
        for key, uri in sources.items():
            if key not in self._streams:
                raise ValueError("{!r} is not a source of this graph; its sources: {}".format(
                    key, ", ".join(self._streams)))
            checked[key] = str(uri)
        return checked

    def render(self, report):
        """The CLI's ``--output`` image for ``report`` -> numpy BGR array.

        Every result drawn onto the report's OWN frame (not the frame most
        recently read), pixel for pixel what the CLI writes for that frame.
        May be called inside this Graph's stream() loop on the same thread;
        from another thread while a call is running it raises RuntimeError
        like any other call. It needs the Graph open: after close() it raises
        RuntimeError("dx_graph.Graph is closed") even for a report taken
        before - render what you need before closing.
        """
        if not isinstance(report, Report):
            raise TypeError("render() takes a dx_graph.Report, got {}"
                            .format(type(report).__name__))
        if self._owner == threading.get_ident():
            # This thread's own stream holds the lock and is suspended at a
            # yield: render touches no executor state, so it runs under the
            # stream's hold.
            core = self._core
            if core is None:
                raise RuntimeError("dx_graph.Graph is closed")
            return core.render(report._r)
        core = self._enter()
        try:
            return core.render(report._r)
        finally:
            self._leave()

    def close(self):
        """Release the executor and every model. Idempotent."""
        core, self._core = self._core, None
        if core is None:
            return
        if self._lock.acquire(False):
            try:
                core.close()
            finally:
                self._lock.release()
        # Otherwise a call on `core` is running on another thread, or a
        # stream on this one is suspended: either holds its own reference.
        # A stream notices at its next step and closes `core` itself; after
        # a run() the models go with the last reference.

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def _enter(self):
        """Take the one-thread lock; return the core it guards. The caller
        opens its try/finally (-> _leave) on the very next statement; from
        the acquire to the return, anything raised - a KeyboardInterrupt
        included - releases the lock here."""
        if self._core is None:
            raise RuntimeError("dx_graph.Graph is closed")
        if not self._lock.acquire(False):
            # Closed first: close() may have landed between the check above
            # and this attempt, while another call still holds the lock.
            if self._core is None:
                raise RuntimeError("dx_graph.Graph is closed")
            if self._owner == threading.get_ident():
                raise RuntimeError(
                    "dx_graph.Graph is already in use by this thread "
                    "(e.g. run() inside a stream loop)")
            raise RuntimeError("dx_graph.Graph is already running in another thread")
        try:
            self._owner = threading.get_ident()
            core = self._core
            if core is None:  # closed while we were acquiring
                raise RuntimeError("dx_graph.Graph is closed")
        except BaseException:
            self._leave()
            raise
        return core

    def _leave(self):
        self._owner = None
        self._lock.release()

    def __repr__(self):
        return "<dx_graph.Graph executor={!r}{}>".format(
            self._executor, "" if self._core is not None else " closed")


_END = object()  # next(frames, _END): the end of a caller's iterator


def _drain_ready(core):
    """Every report that is ready now, in frame order, without waiting."""
    while True:
        r = core.try_next()
        if r is None:
            return
        yield Report(r)


def _require_int(name, value, minimum):
    """An integer (int, numpy integer, ...) of at least `minimum`, as an int."""
    if isinstance(value, bool):
        raise TypeError("{} must be an int, got {!r}".format(name, value))
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError("{} must be an int, got {!r}".format(name, value))
    if number < minimum:
        raise ValueError("{} must be at least {}, got {}".format(name, minimum, number))
    return number
