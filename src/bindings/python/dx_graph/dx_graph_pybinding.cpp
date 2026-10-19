/**
 * @file dx_graph_pybinding.cpp
 * @brief _dx_graph: the C++ core of the dx_graph Python package.
 *
 * Import the dx_graph package, not this module: the package is the
 * supported surface and this module's names may change with it.
 *
 * THE RULES THIS FILE KEEPS
 * -------------------------
 *  - Everything the CLI decides is decided by the CLI's own code: the build
 *    sequence (consumer::PrepareGraph), the executors
 *    (consumer::{Sync,Async}FrameExecutor), the report schema
 *    (consumer::ReportToJson), the order the sources are read in
 *    (consumer::StreamReader). This file only converts.
 *  - numpy <-> cv::Mat by copy, with the GIL held (design P9).
 *  - Every engine call - building the graph, which loads every model, and
 *    every executor call - runs with the GIL released (P10). Nothing inside
 *    a released region touches a Python object; the one exception is the
 *    interrupt hook, which re-acquires the GIL itself (P12).
 *  - One thread at a time per Graph is enforced by the Python layer (P11).
 *  - Nothing is left in flight behind the caller's back: run_frame refuses
 *    while a submitted frame or its report is pending (ruling R2(a)), and
 *    abandon() finishes and discards whatever a stream left (P14).
 *  - Nothing carries over from one run or stream to the next: every node's
 *    tracker is reset at each run_frame and at the first submit of each
 *    stream, so a reused Graph gives what a fresh one gives.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "common/graph/graph_config.hpp"
#include "common/graph/graph_error.hpp"
#include "common/graph/graph_runner_async.hpp"
#include "common/graph/graph_visualizer.hpp"
#include "common/graph/shape.hpp"
#include "common/graph/stage_graph.hpp"
#include "common/base/i_input_source.hpp"
#include "common/registry/static_model_registry.hpp"
#include "multi_model_graph/graph_consumer.hpp"

namespace py = pybind11;

namespace {

using dxapp::graph::AsyncOptions;
using dxapp::graph::BoxesData;
using dxapp::graph::DenseMapData;
using dxapp::graph::FrameData;
using dxapp::graph::FrameReport;
using dxapp::graph::GraphError;
using dxapp::graph::GraphSpec;
using dxapp::graph::ImageData;
using dxapp::InputSourcePtr;
using dxapp::graph::LabelMapData;
using dxapp::graph::Shape;
using dxapp::graph::StageDataPtr;
using dxapp::graph::StageGraph;
using dxapp::graph::StagePorts;
using dxapp::graph::StageResult;
using dxapp::graph::StaticModelRegistry;
namespace consumer = dxapp::graph::consumer;

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// dx_graph.GraphError, a ValueError subclass. Created once in module init
/// and owned by the module for the interpreter's lifetime.
PyObject* g_graph_error = NULL;

void RaiseGraphError(const std::string& code, const std::string& message) {
    py::object type = py::reinterpret_borrow<py::object>(g_graph_error);
    py::object error = type(message);
    error.attr("code") = code;
    PyErr_SetObject(g_graph_error, error.ptr());
}

std::string Strip(const std::string& text) {
    const char* space = " \t\r\n\f\v";
    const std::size_t first = text.find_first_not_of(space);
    if (first == std::string::npos) return std::string();
    const std::size_t last = text.find_last_not_of(space);
    return text.substr(first, last - first + 1);
}

/// GraphError -> dx_graph.GraphError(.code = ToString(code), CLI message).
/// MissingModelsError -> the same class with code "MODEL_MISSING" and the
/// CLI's stderr text, stripped (ruling R1). Other engine errors ->
/// RuntimeError. pybind11's own exceptions (TypeError, ValueError, a Python
/// error already set) are passed on untouched: builtin_exception is itself a
/// std::runtime_error and must not be caught as one.
void TranslateEngineErrors(std::exception_ptr pending) {
    try {
        if (pending) std::rethrow_exception(pending);
    } catch (const py::error_already_set&) {
        throw;
    } catch (const py::builtin_exception&) {
        throw;
    } catch (const GraphError& error) {
        RaiseGraphError(dxapp::graph::ToString(error.code()), error.what());
    } catch (const consumer::MissingModelsError& error) {
        RaiseGraphError("MODEL_MISSING", Strip(error.what()));
    } catch (const std::logic_error& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (const std::runtime_error& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
}

// ---------------------------------------------------------------------
// numpy <-> cv::Mat, always by copy, always with the GIL held
// ---------------------------------------------------------------------

std::string DescribeFrame(const py::handle& frame) {
    if (py::isinstance<py::array>(frame)) {
        py::array array = py::reinterpret_borrow<py::array>(frame);
        return py::str(array.dtype()).cast<std::string>() + " " +
               py::repr(array.attr("shape")).cast<std::string>();
    }
    return py::str(py::type::handle_of(frame).attr("__name__"))
        .cast<std::string>();
}

/// A (H, W, 3) uint8 array of any strides -> a fresh CV_8UC3 Mat. The dtype
/// is checked BEFORE the c_style|forcecast conversion, so a float frame is
/// rejected rather than silently cast.
cv::Mat FrameFromNumpy(const py::object& frame) {
    const std::string rule =
        "frame must be a numpy uint8 array of shape (H, W, 3), got ";
    if (!py::isinstance<py::array>(frame)) {
        throw py::type_error(rule + DescribeFrame(frame));
    }
    py::array array = py::reinterpret_borrow<py::array>(frame);
    const py::dtype dtype = array.dtype();
    if (dtype.kind() != 'u' || dtype.itemsize() != 1) {
        throw py::type_error(rule + DescribeFrame(frame));
    }
    if (array.ndim() != 3 || array.shape(2) != 3 || array.shape(0) <= 0 ||
        array.shape(1) <= 0) {
        throw py::value_error(rule + DescribeFrame(frame));
    }
    typedef py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
        Contiguous;
    // ensure() clears the Python error it hit, so there is nothing to
    // re-raise: say what failed instead (ruling R2(e)).
    Contiguous dense = Contiguous::ensure(array);
    if (!dense) {
        throw py::value_error("could not convert frame to a contiguous uint8 array");
    }
    const int rows = static_cast<int>(dense.shape(0));
    const int cols = static_cast<int>(dense.shape(1));
    cv::Mat mat(rows, cols, CV_8UC3);
    std::memcpy(mat.data, dense.data(),
                static_cast<std::size_t>(rows) * cols * 3);
    return mat;
}

py::dtype DtypeOf(const cv::Mat& mat) {
    switch (mat.depth()) {
        case CV_8U: return py::dtype::of<std::uint8_t>();
        case CV_8S: return py::dtype::of<std::int8_t>();
        case CV_16U: return py::dtype::of<std::uint16_t>();
        case CV_16S: return py::dtype::of<std::int16_t>();
        case CV_32S: return py::dtype::of<std::int32_t>();
        case CV_32F: return py::dtype::of<float>();
        case CV_64F: return py::dtype::of<double>();
        default: break;
    }
    throw std::runtime_error("dx_graph: cannot convert a cv::Mat of type " +
                             std::to_string(mat.type()) + " to numpy");
}

/// Any 2-D Mat of 1-4 channels -> a C-contiguous numpy copy: (rows, cols)
/// for one channel, (rows, cols, ch) otherwise. Rows are copied one by one
/// because a Mat's rows need not be contiguous.
py::array NumpyFromMat(const cv::Mat& mat) {
    const int channels = mat.channels();
    if (mat.dims > 2 || channels < 1 || channels > 4) {
        throw std::runtime_error(
            "dx_graph: cannot convert a cv::Mat with " +
            std::to_string(mat.dims) + " dims and " +
            std::to_string(channels) + " channels to numpy");
    }
    std::vector<py::ssize_t> shape;
    shape.push_back(mat.rows);
    shape.push_back(mat.cols);
    if (channels > 1) shape.push_back(channels);
    py::array out(DtypeOf(mat), shape);
    const std::size_t row_bytes =
        static_cast<std::size_t>(mat.cols) * mat.elemSize();
    unsigned char* target = static_cast<unsigned char*>(out.mutable_data());
    for (int y = 0; y < mat.rows; ++y) {
        std::memcpy(target + y * row_bytes, mat.ptr<unsigned char>(y),
                    row_bytes);
    }
    return out;
}

// ---------------------------------------------------------------------
// _Report
// ---------------------------------------------------------------------

/// One frame's report and the frame it belongs to (P7).
class PyReport {
 public:
    PyReport(const FrameReport& report, const cv::Mat& frame, bool name_stream)
        : report_(report), frame_(frame), name_stream_(name_stream) {}

    std::size_t frame_index() const { return report_.frame_index; }
    const std::string& error() const { return report_.error; }
    const std::string& stream() const { return report_.stream; }
    const FrameReport& report() const { return report_; }
    const cv::Mat& shown() const { return frame_; }

    py::object frame() const {
        if (frame_.empty()) return py::none();
        return NumpyFromMat(frame_);
    }

    std::string json() const { return consumer::ReportToJson(report_, name_stream_).dump(); }

    /// (path, array) for every Mat in the report, where path is the JSON
    /// path, in ReportToJson's own key names, of that Mat's summary object:
    /// ["nodes", "seg", "payload", "labels"],
    /// ["roi_nodes", "attr", 0, "payload", "items", 2, "mask"],
    /// ["nodes", "drive", "ports", "drivable", "labels"], ...
    /// A path is listed exactly when ReportToJson writes that summary.
    py::list arrays() const {
        py::list out;
        typedef std::map<std::string, StageResult>::const_iterator NodeIt;
        for (NodeIt it = report_.node_results.begin();
             it != report_.node_results.end(); ++it) {
            py::list prefix;
            prefix.append("nodes");
            prefix.append(it->first);
            AddPayload(it->second.data, Extended(prefix, "payload"), &out);
            AddPorts(it->second.ports, prefix, &out);
        }
        typedef std::map<std::string, std::vector<StageResult> >::const_iterator
            RoiIt;
        for (RoiIt it = report_.roi_results.begin();
             it != report_.roi_results.end(); ++it) {
            for (std::size_t k = 0; k < it->second.size(); ++k) {
                py::list prefix;
                prefix.append("roi_nodes");
                prefix.append(it->first);
                prefix.append(k);
                AddPayload(it->second[k].data, Extended(prefix, "payload"), &out);
                AddPorts(it->second[k].ports, prefix, &out);
            }
        }
        return out;
    }

 private:
    static void Add(const py::list& prefix, const py::object& key,
                    const cv::Mat& mat, py::list* out) {
        py::list path;
        for (std::size_t i = 0; i < prefix.size(); ++i) path.append(prefix[i]);
        path.append(key);
        out->append(py::make_tuple(path, NumpyFromMat(mat)));
    }

    /// A new list: prefix + [key]. (Assigning a py::list shares the list.)
    static py::list Extended(const py::list& prefix, const char* key) {
        py::list path;
        for (std::size_t i = 0; i < prefix.size(); ++i) path.append(prefix[i]);
        path.append(key);
        return path;
    }

    /// Each non-null port's payload, under [<result prefix>, "ports", name].
    static void AddPorts(const StagePorts& ports, const py::list& prefix,
                         py::list* out) {
        for (StagePorts::const_iterator it = ports.begin(); it != ports.end(); ++it) {
            if (!it->second) continue;
            py::list path = Extended(prefix, "ports");
            path.append(it->first);
            AddPayload(it->second, path, out);
        }
    }

    /// Mirrors PayloadToJson in graph_consumer.cpp: the same shapes, the same
    /// keys, the same "mask only when non-empty" rule.
    static void AddPayload(const StageDataPtr& data, const py::list& prefix,
                           py::list* out) {
        if (!data) return;
        switch (data->shape()) {
            case Shape::kFrame:
                Add(prefix, py::str("image"),
                    static_cast<const FrameData*>(data.get())->image, out);
                break;
            case Shape::kBoxes:
            case Shape::kObBoxes:
            case Shape::kInstances: {
                const BoxesData* boxes =
                    static_cast<const BoxesData*>(data.get());
                for (std::size_t i = 0; i < boxes->items.size(); ++i) {
                    if (boxes->items[i].mask.empty()) continue;
                    py::list path;
                    for (std::size_t p = 0; p < prefix.size(); ++p) {
                        path.append(prefix[p]);
                    }
                    path.append("items");
                    path.append(i);
                    Add(path, py::str("mask"), boxes->items[i].mask, out);
                }
                break;
            }
            case Shape::kLabelMap:
                Add(prefix, py::str("labels"),
                    static_cast<const LabelMapData*>(data.get())->labels, out);
                break;
            case Shape::kDenseMap:
                Add(prefix, py::str("values"),
                    static_cast<const DenseMapData*>(data.get())->values, out);
                break;
            case Shape::kImage:
                Add(prefix, py::str("image"),
                    static_cast<const ImageData*>(data.get())->image, out);
                break;
            default:
                break;  // no dense payload
        }
    }

    FrameReport report_;
    cv::Mat frame_;
    bool name_stream_;  ///< the JSON names the stream: the graph has several
};

/// Runs `drop` without the GIL when this thread holds it - destroying a
/// graph waits for every stage's outstanding jobs. Raw C API rather than
/// gil_scoped_release, because a destructor may run while the interpreter
/// is finalizing.
template <typename Drop>
void WithoutGil(Drop drop) {
    if (!PyGILState_Check()) {
        drop();
        return;
    }
    PyThreadState* saved = PyEval_SaveThread();
    drop();
    PyEval_RestoreThread(saved);
}

// ---------------------------------------------------------------------
// _Source
// ---------------------------------------------------------------------

/// The graph's input, opened by the CLI's rule (consumer::OpenGraphSource).
/// Frames are read into a C++ buffer and submitted from there: a
/// source-driven stream never makes a numpy round trip (design section 5).
///
/// Each frame is read into a fresh buffer. The sync executor's report holds
/// the source node's frame without copying it, and a Report here outlives
/// the next read. The video reader writes into a buffer of the same size in
/// place, so a reused buffer would carry the newest frame into every report
/// still held.
class PySource {
 public:
    PySource(InputSourcePtr source, const std::string& uri)
        : source_(std::move(source)), uri_(uri), frames_read_(0) {}

    ~PySource() {
        InputSourcePtr source = std::move(source_);
        WithoutGil([&] { source.reset(); });
    }

    /// Next frame into frame(), a buffer of its own; false at the end of
    /// the input. Called with the GIL released.
    bool Read() {
        frame_.release();  // never the buffer a report may still hold
        if (!source_->getFrame(frame_)) return false;
        ++frames_read_;
        return true;
    }

    const cv::Mat& frame() const { return frame_; }
    std::size_t frames_read() const { return frames_read_; }
    const std::string& uri() const { return uri_; }

 private:
    InputSourcePtr source_;
    std::string uri_;
    cv::Mat frame_;  ///< the frame just read; a new buffer for every read
    std::size_t frames_read_;
};

// ---------------------------------------------------------------------
// _Sources
// ---------------------------------------------------------------------

/// The graph's sources, read in turn through the CLI's own reader
/// (consumer::StreamReader), so dx_graph's sources= interleaves exactly as
/// the CLI does. Frames go from its buffer straight into the executor.
///
/// Each frame is read into a fresh buffer, for PySource's reason. Here a
/// reused buffer is overwritten when one stream is read twice in a row,
/// as happens once the other streams have ended.
class PySources {
 public:
    explicit PySources(std::unique_ptr<consumer::StreamReader> reader)
        : reader_(std::move(reader)) {}

    ~PySources() {
        std::unique_ptr<consumer::StreamReader> reader = std::move(reader_);
        WithoutGil([&] { reader.reset(); });
    }

    /// The next stream's next frame into frame(), a buffer of its own; false
    /// once every stream has ended. Called with the GIL released.
    bool Read(std::size_t* stream) {
        frame_.release();  // never the buffer a report may still hold
        std::size_t index = 0;
        return reader_->Next(stream, &frame_, &index);
    }

    const cv::Mat& frame() const { return frame_; }

    /// Spec R9, as the CLI: once every stream has run to its end, one
    /// GraphError (GRAPH_SCHEMA) for the streams that ended without a frame,
    /// its message the CLI's "produced no frames" line for each, in
    /// declaration order, joined by newlines. Nothing when there is none.
    /// Called with the GIL held.
    void RaiseIfEmpty() const {
        std::string message;
        for (std::size_t s = 0; s < reader_->size(); ++s) {
            if (!reader_->ended(s) || reader_->frames_read(s) != 0) continue;
            if (!message.empty()) message += "\n";
            message += consumer::NoFramesError(reader_->input(s).uri).what();
        }
        if (message.empty()) return;
        RaiseGraphError(dxapp::graph::ToString(dxapp::graph::GraphErrorCode::kGraphSchema),
                        message);
        throw py::error_already_set();
    }

 private:
    std::unique_ptr<consumer::StreamReader> reader_;
    cv::Mat frame_;  ///< the frame just read; a new buffer for every read
};

// ---------------------------------------------------------------------
// _Graph
// ---------------------------------------------------------------------

/// The interrupt hook's state. Every field is touched only by the thread
/// running the engine call - the hook runs on the executor's driver, which
/// is that thread. `kept` is set and reset, and `checks` counted, only with
/// the GIL held; the hook reads whether `kept` is set without it.
struct InterruptState {
    typedef std::chrono::steady_clock Clock;

    /// The signal's Python error, from the check that saw it until the
    /// engine call returns and raises it.
    std::unique_ptr<py::error_already_set> kept;
    /// When the hook last took the GIL to check. Kept across engine calls,
    /// so the rate holds however a stream splits its waits between calls.
    Clock::time_point last_check;
    /// Checks made, for the tests (_Graph._interrupt_checks).
    std::size_t checks;

    InterruptState() : checks(0) {}
};

/// D1: how often the hook may take the GIL while the executor waits. The
/// executor asks after every quiet wait slice (5 ms); re-acquiring the GIL
/// that often competes with every other Python thread for nothing, while
/// a Ctrl-C a person presses is served just as well within 50 ms.
const InterruptState::Clock::duration kInterruptCheckInterval =
    std::chrono::milliseconds(50);

// ---------------------------------------------------------------------
// Test-only: a graph whose one model never completes
// ---------------------------------------------------------------------
//
// _stalled_graph_for_tests() builds a real _Graph - the same hooks, the same
// async executor, the same CallEngine - over the registry below, so a test
// can wait on a stalled stage without hardware and watch the interrupt hook
// work: its throttle, a signal it catches, the stuck-stage line on stderr.

/// A job the runtime lost: submit() drops the callback, so nothing ever
/// completes, and flush() and destruction have nothing to wait for.
class NeverCompletesStage : public dxapp::graph::IStage {
 public:
    StageResult run(const dxapp::graph::StageInput&) {
        throw std::runtime_error("never_completes: a stage for async tests only");
    }
    void submit(const dxapp::graph::StageInput&, dxapp::graph::StageCallback) {}
    void flush() {}
    Shape outputShape() const { return Shape::kBoxes; }
    dxapp::graph::InputContract inputContract() const {
        return dxapp::graph::InputContract::kFullFrame;
    }
};

/// One model, "never_completes", whose stages are NeverCompletesStage.
class NeverCompletesRegistry : public dxapp::graph::IModelRegistry {
 public:
    NeverCompletesRegistry() {
        info_.model_name = "never_completes";
        info_.task = "object_detection";
        info_.dxnn_file = "never_completes.dxnn";
        info_.input_width = 64;
        info_.input_height = 48;
        info_.output_shape = Shape::kBoxes;
        info_.input_contract = dxapp::graph::InputContract::kFullFrame;
        info_.ready = true;
    }
    const dxapp::graph::ModelInfo* find(const std::string& name) const {
        return name == info_.model_name ? &info_ : NULL;
    }
    std::vector<dxapp::graph::ModelInfo> list() const {
        return std::vector<dxapp::graph::ModelInfo>(1, info_);
    }
    std::unique_ptr<dxapp::graph::IStage> createStage(
        const std::string&, const std::string&,
        const dxapp::graph::StageParams&) const {
        return std::unique_ptr<dxapp::graph::IStage>(new NeverCompletesStage());
    }

 private:
    dxapp::graph::ModelInfo info_;  // stable address: find() hands it out
};

const char kStalledGraphJson[] =
    "{\"version\": 1, \"name\": \"stalled\", \"nodes\": ["
    "{\"id\": \"cam\", \"type\": \"source\", \"uri\": \"unused.jpg\"},"
    "{\"id\": \"stuck\", \"model\": \"never_completes\"}],"
    "\"edges\": [{\"from\": \"cam\", \"to\": \"stuck\"}]}";

class PyGraph {
 public:
    PyGraph(const std::string& path, const py::object& model_dir,
            const std::string& executor, std::size_t max_frames_in_flight,
            std::size_t max_jobs_per_stage, std::size_t stall_timeout_ms)
        : registry_(new StaticModelRegistry()),
          executor_name_(executor),
          interrupt_(std::make_shared<InterruptState>()),
          pending_(0) {
        Configure(max_frames_in_flight, max_jobs_per_stage, stall_timeout_ms);
        // The executor first: it validates its options and costs nothing,
        // whereas the graph loads every model.
        std::unique_ptr<consumer::FrameExecutor> ready = MakeExecutor();

        const std::string models = model_dir.is_none()
                                       ? consumer::DefaultModelDir()
                                       : model_dir.cast<std::string>();
        std::unique_ptr<StageGraph> graph(new StageGraph());
        {
            py::gil_scoped_release release;
            consumer::PrepareGraph(path, models, *registry_, &spec_, graph.get());
        }
        // Nothing above left a half-built object behind: a throw from
        // PrepareGraph destroys `graph` here and pybind11 never binds this.
        graph_ = std::move(graph);
        next_index_.assign(graph_->streams().size(), 0);
        executor_ = std::move(ready);
    }

    struct StalledForTests {};

    /// Test-only (_stalled_graph_for_tests): an async Graph over
    /// kStalledGraphJson and NeverCompletesRegistry, with the default
    /// options but `stall_timeout_ms`. Everything else is this class's own.
    PyGraph(StalledForTests, std::size_t stall_timeout_ms)
        : registry_(new NeverCompletesRegistry()),
          executor_name_("async"),
          interrupt_(std::make_shared<InterruptState>()),
          pending_(0) {
        Configure(AsyncOptions().max_frames_in_flight,
                  AsyncOptions().max_jobs_per_stage, stall_timeout_ms);
        std::unique_ptr<consumer::FrameExecutor> ready = MakeExecutor();
        spec_ = dxapp::graph::ParseGraphText(kStalledGraphJson, "<stalled test graph>");
        dxapp::graph::ValidateGraph(spec_, *registry_);
        std::unique_ptr<StageGraph> graph(new StageGraph());
        graph->Build(spec_, *registry_, "", false);
        graph_ = std::move(graph);
        next_index_.assign(graph_->streams().size(), 0);
        executor_ = std::move(ready);
    }

    /// As Close(): the executor, then the graph, without the GIL (R2(c)).
    ~PyGraph() {
        std::unique_ptr<consumer::FrameExecutor> executor = std::move(executor_);
        std::unique_ptr<StageGraph> graph = std::move(graph_);
        WithoutGil([&] {
            executor.reset();
            graph.reset();
        });
    }

    const std::string& executor() const { return executor_name_; }

    /// The async executor's options; empty for "sync", which has none -
    /// the values a sync Graph was given are validated and then unused.
    py::dict options() const {
        py::dict out;
        if (executor_name_ != "async") return out;
        out["max_frames_in_flight"] = options_.max_frames_in_flight;
        out["max_jobs_per_stage"] = options_.max_jobs_per_stage;
        out["stall_timeout_ms"] = options_.stall_timeout_ms;
        return out;
    }

    /// The source node ids, in declaration order: one stream each.
    py::list stream_ids() const {
        RequireOpen();
        py::list out;
        const std::vector<dxapp::graph::StreamPlan>& streams = graph_->streams();
        for (std::size_t s = 0; s < streams.size(); ++s) out.append(streams[s].source_id);
        return out;
    }

    /// One frame of stream `stream`, start to finish, as frame 0 - the
    /// index the CLI gives a single image, and on a fresh tracker, as the
    /// CLI runs a single image: a run is independent of every run and
    /// stream before it.
    PyReport RunFrame(const py::object& frame, std::size_t stream) {
        RequireOpen();
        RequireStream(stream);
        // Otherwise TryNext would hand out a stream's leftover report as this
        // frame's (ruling R2(a)).
        if (pending_ != 0) {
            throw std::runtime_error(
                "dx_graph.Graph has reports in flight; finish the stream first");
        }
        const cv::Mat image = FrameFromNumpy(frame);  // GIL held
        FrameReport report;
        cv::Mat shown;
        bool taken = false;
        CallEngine([&] {
            ResetTrackers();  // nothing is in flight: pending_ is 0
            executor_->Submit(*graph_, stream, image, 0);
            executor_->Finish(*graph_);
            taken = executor_->TryNext(&report, &shown);
        });
        if (!taken) {
            throw std::logic_error("dx_graph: the frame finished without a report");
        }
        return PyReport(report, shown, consumer::NamesStreams(*graph_));
    }

    // --- streaming (design section 5; the Python layer's stream()) -----
    //
    // Each stream's frames are numbered 0, 1, ... from the first submit
    // after finish() or abandon(): each stream is numbered as the CLI numbers
    // its source, and a sequence starts on fresh trackers, as the CLI starts
    // on its sources.

    /// Admit one numpy frame of stream `stream` (copied under the GIL).
    void Submit(const py::object& frame, std::size_t stream) {
        RequireOpen();
        RequireStream(stream);
        const cv::Mat image = FrameFromNumpy(frame);  // GIL held
        CallEngine([&] { SubmitReleased(stream, image); });
    }

    /// Read the source's next frame and admit it, all inside C++. False at
    /// the end of the input; a source that ends before its first frame is
    /// the CLI's "produced no frames" GraphError.
    bool SubmitFromSource(PySource* source) {
        RequireOpen();
        bool read = false;
        CallEngine([&] {
            read = source->Read();
            if (read) SubmitReleased(graph_->OnlyStream("dx_graph"), source->frame());
        });
        if (!read && source->frames_read() == 0) {
            throw consumer::NoFramesError(source->uri());
        }
        return read;
    }

    /// Read the next stream's frame and admit it, all inside C++. False once
    /// every stream has ended. A stream that ends without a frame does not
    /// stop the others: it is reported once they have all ended, by
    /// PySources::RaiseIfEmpty (spec R9).
    bool SubmitFromSources(PySources* sources) {
        RequireOpen();
        bool read = false;
        CallEngine([&] {
            std::size_t stream = 0;
            read = sources->Read(&stream);
            if (read) SubmitReleased(stream, sources->frame());
        });
        return read;
    }

    /// The next report in frame order, with its own frame, or None.
    std::unique_ptr<PyReport> TryNext() {
        RequireOpen();
        FrameReport report;
        cv::Mat shown;
        bool taken = false;
        CallEngine([&] {
            taken = executor_->TryNext(&report, &shown);
            if (taken) --pending_;
        });
        if (!taken) return std::unique_ptr<PyReport>();
        return std::unique_ptr<PyReport>(
            new PyReport(report, shown, consumer::NamesStreams(*graph_)));
    }

    /// Wait for every admitted frame; their reports stay for try_next().
    void Finish() {
        RequireOpen();
        CallEngine([&] {
            executor_->Finish(*graph_);
            next_index_.assign(next_index_.size(), 0);
        });
    }

    /// P14: finish what is in flight and discard every report left, so the
    /// next run_frame or stream starts from nothing. A closed graph has
    /// nothing to abandon.
    void Abandon() {
        if (!graph_ || !executor_) return;
        CallEngine([&] {
            executor_->Finish(*graph_);
            FrameReport report;
            while (executor_->TryNext(&report, NULL)) {
            }
            pending_ = 0;
            next_index_.assign(next_index_.size(), 0);
        });
    }

    /// `uri` None: the graph's own source; otherwise it overrides the source
    /// node's "uri", exactly as the CLI's --input does.
    std::unique_ptr<PySource> OpenSource(const py::object& uri) {
        RequireOpen();
        const std::string override_uri = uri.is_none() ? std::string()
                                                       : uri.cast<std::string>();
        std::string chosen;
        InputSourcePtr source;
        {
            py::gil_scoped_release release;
            source = consumer::OpenGraphSource(spec_, override_uri, &chosen);
        }
        return std::unique_ptr<PySource>(new PySource(std::move(source), chosen));
    }

    /// Every source of the graph, opened by the CLI's rule for each stream
    /// (consumer::ResolveStreams, consumer::OpenStream): `sources` maps a
    /// source id to the uri that overrides its "uri", as the CLI's
    /// --input <id>=<uri>; a source left out reads its own. `limit` is the
    /// CLI's --frames, per stream; 0 = all.
    std::unique_ptr<PySources> OpenSources(const py::dict& sources, std::size_t limit) {
        RequireOpen();
        std::map<std::string, std::string> overrides;
        for (auto item : sources) {
            overrides[py::cast<std::string>(item.first)] = py::cast<std::string>(item.second);
        }
        std::vector<consumer::StreamUri> uris;
        const std::string problem = consumer::ResolveStreams(spec_, overrides, &uris);
        if (!problem.empty()) throw py::value_error(problem);
        std::unique_ptr<consumer::StreamReader> reader;
        {
            py::gil_scoped_release release;
            std::vector<consumer::StreamInput> inputs;
            for (std::size_t s = 0; s < uris.size(); ++s) {
                consumer::StreamInput input;
                input.source = uris[s].source;
                input.uri = uris[s].uri;
                input.input = consumer::OpenStream(spec_, uris[s]);
                inputs.push_back(std::move(input));
            }
            reader.reset(new consumer::StreamReader(std::move(inputs), limit));
        }
        return std::unique_ptr<PySources>(new PySources(std::move(reader)));
    }

    /// Releases the executor, then the graph (every stage waits for its own
    /// outstanding jobs as it is destroyed - hence without the GIL).
    void Close() {
        std::unique_ptr<consumer::FrameExecutor> executor = std::move(executor_);
        std::unique_ptr<StageGraph> graph = std::move(graph_);
        py::gil_scoped_release release;
        executor.reset();
        graph.reset();
    }

    bool closed() const { return !graph_; }

    /// How many times the interrupt hook has taken the GIL to check for a
    /// signal (tests of the D1 throttle).
    std::size_t interrupt_checks() const { return interrupt_->checks; }

    /// P8: the CLI's --output rendering of `report` - RenderReport onto the
    /// report's OWN frame, exactly the call graph_cli.cpp makes - as a numpy
    /// BGR array; None for a report without a frame. Touches no executor
    /// state, so it may run between a stream's reports.
    py::object Render(const PyReport& report) {
        RequireOpen();
        cv::Mat rendered;
        {
            py::gil_scoped_release release;
            rendered = dxapp::graph::RenderReport(report.shown(), report.report(),
                                                  dxapp::graph::RenderOptionsFor(spec_));
        }
        if (rendered.empty()) return py::none();
        return NumpyFromMat(rendered);
    }

 private:
    /// The executor's options and the P12 hooks. Validates the executor name.
    void Configure(std::size_t max_frames_in_flight, std::size_t max_jobs_per_stage,
                   std::size_t stall_timeout_ms) {
        if (executor_name_ != "sync" && executor_name_ != "async") {
            throw py::value_error("executor must be \"sync\" or \"async\", got \"" +
                                  executor_name_ + "\"");
        }
        options_.max_frames_in_flight = max_frames_in_flight;
        options_.max_jobs_per_stage = max_jobs_per_stage;
        options_.stall_timeout_ms = stall_timeout_ms;
        std::shared_ptr<InterruptState> state = interrupt_;
        // P12. Called by the async executor's driver - this thread, with the
        // GIL released - after every wait slice in which nothing completed.
        // Once a signal is kept the answer stays "interrupted" without the
        // GIL; otherwise the GIL is taken, and the signals checked, at most
        // once per kInterruptCheckInterval (D1).
        options_.interrupted = [state]() -> bool {
            if (state->kept) return true;
            const InterruptState::Clock::time_point now = InterruptState::Clock::now();
            if (now - state->last_check < kInterruptCheckInterval) return false;
            state->last_check = now;
            py::gil_scoped_acquire gil;
            ++state->checks;
            if (PyErr_CheckSignals() != 0) {
                state->kept.reset(new py::error_already_set());
                return true;
            }
            return false;
        };
        options_.on_stuck = [](const std::vector<std::string>& ids) {
            std::fprintf(stderr, "%s\n", consumer::StuckMessage(ids).c_str());
        };
    }

    /// Called with the GIL released, inside CallEngine. The first submit of
    /// a sequence, with nothing in flight, starts every tracker afresh.
    void SubmitReleased(std::size_t stream, const cv::Mat& image) {
        bool fresh = pending_ == 0;
        for (std::size_t s = 0; s < next_index_.size(); ++s) {
            if (next_index_[s] != 0) fresh = false;
        }
        if (fresh) ResetTrackers();
        executor_->Submit(*graph_, stream, image, next_index_[stream]);
        ++next_index_[stream];
        ++pending_;
    }

    void RequireStream(std::size_t stream) const {
        if (stream >= graph_->streams().size()) {
            throw py::index_error("dx_graph: no stream " + std::to_string(stream));
        }
    }

    /// Every tracked node's IouTracker back to no tracks and id 0. Only with
    /// nothing in flight: no stage is applying track ids at the same time.
    void ResetTrackers() {
        std::vector<dxapp::graph::NodeRuntime>& nodes = graph_->mutable_nodes();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (std::size_t s = 0; s < nodes[i].trackers.size(); ++s) {
                if (nodes[i].trackers[s]) nodes[i].trackers[s]->reset();
            }
        }
    }

    std::unique_ptr<consumer::FrameExecutor> MakeExecutor() const {
        if (executor_name_ == "async") {
            return std::unique_ptr<consumer::FrameExecutor>(
                new consumer::AsyncFrameExecutor(options_));
        }
        return std::unique_ptr<consumer::FrameExecutor>(
            new consumer::SyncFrameExecutor());
    }

    void RequireOpen() const {
        if (!graph_ || !executor_) {
            throw std::runtime_error("dx_graph.Graph is closed");
        }
    }

    /// Runs `call` with the GIL released. Afterwards (GIL held again) a
    /// signal the interrupt hook caught is raised - KeyboardInterrupt, as in
    /// the CLI after its drain - in place of whatever `call` threw. Anything
    /// thrown out of `call` replaces the executor: after an exception it may
    /// hold frames whose jobs will never be delivered (graph_runner_async.hpp);
    /// with it go its pending reports, so the count and the numbering start
    /// again. That includes a throw that is not the executor's: `call` in
    /// SubmitFromSource also reads the source, and a read that throws (an
    /// OpenCV exception, an allocation failure) drops the stream's frames in
    /// flight and their reports just the same. A read that merely fails is
    /// no throw - IInputSource::getFrame returns false, the end of the
    /// input - and a source that produced no frames at all is reported after
    /// the call, with the executor kept.
    template <typename Call>
    void CallEngine(Call call) {
        interrupt_->kept.reset();
        try {
            py::gil_scoped_release release;
            call();
        } catch (...) {
            executor_ = MakeExecutor();
            pending_ = 0;
            next_index_.assign(next_index_.size(), 0);
            RaiseKept();
            throw;
        }
        RaiseKept();
    }

    void RaiseKept() {
        if (!interrupt_->kept) return;
        py::error_already_set error = *interrupt_->kept;
        interrupt_->kept.reset();
        throw error;
    }

    // Declaration order is destruction order in reverse: the registry
    // outlives the graph built from it, the graph outlives the executor.
    std::unique_ptr<const dxapp::graph::IModelRegistry> registry_;
    GraphSpec spec_;
    AsyncOptions options_;
    std::string executor_name_;
    std::shared_ptr<InterruptState> interrupt_;
    std::unique_ptr<StageGraph> graph_;
    std::unique_ptr<consumer::FrameExecutor> executor_;
    std::size_t pending_;  ///< submitted, report not yet taken
    std::vector<std::size_t> next_index_;  ///< per stream: the next submitted frame's index
};

}  // namespace

PYBIND11_MODULE(_dx_graph, m) {
    m.doc() = "C++ core of dx_graph; use the dx_graph package, not this module.";
    m.def("build_info", [] {
        py::dict info;
        info["python"] = std::string(DX_GRAPH_PYTHON_VERSION);
        info["project_root"] = std::string(PROJECT_ROOT_DIR);
        return info;
    });

    // The new reference PyErr_NewException returns is g_graph_error's own,
    // kept for the interpreter's lifetime; add_object takes the module
    // attribute's reference itself.
    g_graph_error = PyErr_NewException("dx_graph.GraphError", PyExc_ValueError, NULL);
    if (g_graph_error == NULL) throw py::error_already_set();
    if (PyObject_SetAttrString(g_graph_error, "code", Py_None) != 0) {
        throw py::error_already_set();
    }
    const py::str doc(
        "A graph that cannot be built: .code is the CLI's error code "
        "(e.g. \"GRAPH_EDGE\", \"MODEL_MISSING\"), str() the CLI's message.");
    if (PyObject_SetAttrString(g_graph_error, "__doc__", doc.ptr()) != 0) {
        throw py::error_already_set();
    }
    m.add_object("GraphError", py::handle(g_graph_error));
    // Local: only exceptions leaving this module's own functions (R2(d)).
    py::register_local_exception_translator(&TranslateEngineErrors);

    py::class_<PyReport>(m, "_Report")
        .def_property_readonly("frame_index", &PyReport::frame_index)
        .def_property_readonly("error", &PyReport::error)
        .def("frame", &PyReport::frame)
        .def("json", &PyReport::json)
        .def("arrays", &PyReport::arrays)
        .def_property_readonly("stream", &PyReport::stream);

    py::class_<PySource>(m, "_Source")
        .def("submit_next",
             [](PySource& source, PyGraph& graph) {
                 return graph.SubmitFromSource(&source);
             },
             py::arg("graph"));

    py::class_<PySources>(m, "_Sources")
        .def("submit_next",
             [](PySources& sources, PyGraph& graph) {
                 return graph.SubmitFromSources(&sources);
             },
             py::arg("graph"))
        .def("raise_if_empty", &PySources::RaiseIfEmpty);

    py::class_<PyGraph>(m, "_Graph")
        .def(py::init<const std::string&, const py::object&, const std::string&,
                      std::size_t, std::size_t, std::size_t>(),
             py::arg("path"), py::arg("model_dir"), py::arg("executor"),
             py::arg("max_frames_in_flight"), py::arg("max_jobs_per_stage"),
             py::arg("stall_timeout_ms"))
        .def("run_frame", &PyGraph::RunFrame, py::arg("frame"), py::arg("stream") = 0)
        .def("submit", &PyGraph::Submit, py::arg("frame"), py::arg("stream") = 0)
        .def("try_next", &PyGraph::TryNext)
        .def("finish", &PyGraph::Finish)
        .def("abandon", &PyGraph::Abandon)
        .def("open_source", &PyGraph::OpenSource, py::arg("uri") = py::none())
        .def("open_sources", &PyGraph::OpenSources, py::arg("sources"), py::arg("limit"))
        .def("render", &PyGraph::Render, py::arg("report"))
        .def("close", &PyGraph::Close)
        .def_property_readonly("closed", &PyGraph::closed)
        .def_property_readonly("executor", &PyGraph::executor)
        .def_property_readonly("options", &PyGraph::options)
        .def_property_readonly("stream_ids", &PyGraph::stream_ids)
        .def_property_readonly("_interrupt_checks", &PyGraph::interrupt_checks);

    // Test-only, not part of the dx_graph package: see NeverCompletesStage.
    // stall_timeout_ms 0 means "wait forever", and nothing here ever
    // completes: such a graph would hang its caller, so it is refused.
    m.def("_stalled_graph_for_tests",
          [](std::size_t stall_timeout_ms) {
              if (stall_timeout_ms == 0) {
                  throw py::value_error(
                      "_stalled_graph_for_tests: stall_timeout_ms must be > 0 "
                      "(0 waits forever on a stage that never completes)");
              }
              return std::unique_ptr<PyGraph>(
                  new PyGraph(PyGraph::StalledForTests(), stall_timeout_ms));
          },
          py::arg("stall_timeout_ms"));
}
