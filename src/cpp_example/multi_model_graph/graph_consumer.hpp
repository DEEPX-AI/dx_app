/**
 * @file graph_consumer.hpp
 * @brief What every consumer of the graph engine shares: one report schema,
 *        one path rule, one build sequence.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The engine has two consumers: the multi_model_graph_{sync,async} CLI and
 * the dx_graph Python module (src/bindings/python/dx_graph/). They must
 * agree byte for byte on the JSON a FrameReport becomes, on where a
 * relative "uri" is looked up, on which checks run before a graph is built
 * and what those checks say when they fail. Two copies of that code would
 * drift, and the drift would only show as a Python result that no longer
 * matches the CLI's. So it lives here, once, moved out of graph_cli.cpp
 * unchanged, and both consumers call it.
 *
 * WHY IT IS NOT UNDER common/graph/
 * ---------------------------------
 * Same reason as graph_cli.hpp: this is consumer code. It includes the
 * concrete registry's neighbourhood (the input factory) and is linked
 * next to StaticModelRegistry, which
 * scripts/check_graph_boundary.py keeps out of every file under
 * common/graph/. Living here keeps that guard meaningful.
 */
#ifndef DXAPP_MULTI_MODEL_GRAPH_GRAPH_CONSUMER_HPP
#define DXAPP_MULTI_MODEL_GRAPH_GRAPH_CONSUMER_HPP

#include <cmath>
#include <cstddef>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/base/i_input_source.hpp"
#include "common/graph/graph_config.hpp"
#include "common/graph/graph_error.hpp"
#include "common/graph/graph_runner_async.hpp"
#include "common/graph/graph_runner_sync.hpp"
#include "common/graph/i_registry.hpp"
#include "common/graph/stage_graph.hpp"
#include "common/third_party/nlohmann_json.hpp"

namespace dxapp {
namespace graph {
namespace consumer {

/// The root repository-relative paths are read against: RuntimeRoot()
/// (common/utility/repo_path.hpp), the build's repository unless $DXAPP_ROOT
/// says otherwise or the program is an exported project.
std::string ProjectRoot();

/// True when `path` can be opened for reading.
bool FileExists(const std::string& path);

/// Where the .dxnn files live unless told otherwise:
/// ProjectRoot() + "/assets/models".
std::string DefaultModelDir();

// ---------------------------------------------------------------------
// The FrameReport as JSON
// ---------------------------------------------------------------------

/**
 * @brief One frame's report.
 *
 * Deliberately carries nothing executor-specific - no timings, no
 * executor name - because tests/cpp_example/test_graph_cli.py compares the
 * sync binary's file with the async binary's byte for byte. Ordering is
 * already canonical: node_results and roi_results are std::maps (keyed by
 * node id) and each roi_results vector was stable_sorted by ByOrigin -
 * (parent_node, parent_index, roi_index) - by whichever executor produced
 * it.
 *
 * Non-finite numbers (U-11): nlohmann would write NaN, +Inf and -Inf all as
 * null. Every non-finite float in the tree is instead the STRING "NaN",
 * "Infinity" or "-Infinity" - valid JSON for strict parsers (JSON.parse,
 * jq, nlohmann::json::parse), distinct from each other and from null, and
 * what Python's float() and JavaScript's Number() convert back. A report
 * holds no null at all. Finite numbers are untouched, so a finite report's
 * bytes are what they always were. dx_graph's Report.to_dict() is
 * json.loads of this same text, so it returns the same strings.
 *
 * Streams (SP2 R4): with `name_stream` it adds "stream": report.stream.
 * Every call site decides; pass NamesStreams(graph), so a one-source
 * report is byte-identical to what it was before streams existed.
 *
 * Timing: with `with_timing` it adds "node_ms": {"<node id>": ms, ...},
 * FrameReport::node_ms rounded to 0.001 ms. Off by default, so a report
 * stays free of anything executor-specific (--report-timing turns it on).
 *
 * Events: a frame whose logic nodes raised events gets "events": [{"node",
 * "event", "track_id", "time", "data"}], in FrameReport::events order. A
 * frame without events has no key, so a graph without logic nodes writes
 * exactly the bytes it always did.
 */
nlohmann::json ReportToJson(const FrameReport& report, bool name_stream,
                            bool with_timing = false);

/// True when a graph's reports name their stream: it has more than one source.
bool NamesStreams(const StageGraph& graph);

/// A number as a person writes it: a whole number without ".0", a
/// non-finite one as "NaN", "Infinity" or "-Infinity" (U-11). For event
/// data and --list-ops; report payloads keep their own encoding.
template <typename Json>
Json JsonNumber(double value) {
    if (std::isnan(value)) return Json("NaN");
    if (std::isinf(value)) return Json(value > 0 ? "Infinity" : "-Infinity");
    if (value == std::floor(value) && std::fabs(value) < 1e15) {
        return Json(static_cast<long long>(value));
    }
    return Json(value);
}

/**
 * @brief One event as the CLI prints it after "[EVENT] " and writes it to
 *        --events: {"frame", "time", "stream", "node", "event", "track_id",
 *        "data"}, in that order.
 *
 * "time" is report.time_s rounded to 0.001 s, "track_id" -1 when the event
 * is about no one track, "data" the event's numbers then its text, flat.
 */
nlohmann::ordered_json EventLineJson(const GraphEvent& event, const FrameReport& report);

// ---------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------

/// A relative "uri" in a graph file is relative to the repository, not to
/// the caller's working directory: the sample graphs ship with
/// "sample/img/..." and must run from anywhere.
std::string ResolvePath(const std::string& path);

/// rtsp://..., camera:<N>, or a file path (through ResolvePath).
InputSourcePtr OpenInput(const std::string& uri);

/// The uri a run reads: `override_uri` (the CLI's --input) when it is not
/// empty, else the first source node's "uri"; "" when neither exists.
std::string SourceUri(const GraphSpec& spec, const std::string& override_uri);

/**
 * @brief The CLI's one-source rule, whole: SourceUri, then OpenStream for
 *        the first source node.
 *
 * Throws GraphError(kGraphSchema) with the CLI's message when there is no
 * uri, or when the uri cannot be opened (OpenInput's own reason becomes the
 * "what happened"). *uri receives the uri chosen, before anything can
 * throw, so a caller can name it in a later message.
 */
InputSourcePtr OpenGraphSource(const GraphSpec& spec,
                               const std::string& override_uri,
                               std::string* uri);

/// What the CLI reports when the source opened but yielded no frame.
GraphError NoFramesError(const std::string& uri);

// ---------------------------------------------------------------------
// Streams: one per source node (SP2)
// ---------------------------------------------------------------------

struct StreamUri {
    std::string source;  ///< the source node's id
    std::string uri;     ///< what it reads
};

/// The CLI's --input values into *overrides (cleared first; left empty on
/// an error). A value is "<source_id>=<uri>" only when it begins with a
/// source node's id followed by '=' (the longest such id, since an id may
/// itself hold '='); anything else is a bare uri for the graph's only
/// source (so "rtsp://h/s?a=1" stays a uri). With several
/// sources a bare uri is an error, and "camX=a.mp4" is reported as naming
/// no source node. "" on success (and when the graph has no source:
/// ValidateGraph reports that), otherwise the usage error.
std::string ParseInputOverrides(const GraphSpec& spec, const std::vector<std::string>& inputs,
                                std::map<std::string, std::string>* overrides);

/// Every source node, in declaration order, with the uri it reads: its
/// override, else its own "uri". "" on success, otherwise the problem (an
/// override naming no source node).
std::string ResolveStreams(const GraphSpec& spec,
                           const std::map<std::string, std::string>& overrides,
                           std::vector<StreamUri>* streams);

/// OpenInput for one stream, with the CLI's GraphError(kGraphSchema) for a
/// missing uri or one that will not open. One source: the text
/// OpenGraphSource always printed. Several: the hint names the stream.
InputSourcePtr OpenStream(const GraphSpec& spec, const StreamUri& stream);

struct StreamInput {
    std::string source;
    std::string uri;
    InputSourcePtr input;
};

/**
 * The graph's streams read in turn (SP2 R1): one frame from each stream
 * still reading, in source declaration order, round after round. A stream
 * ends when its source gives no frame, or when it has given
 * frames_per_stream frames (0: no limit); an ended stream is never read
 * again. The CLI and dx_graph both read through this class, so they cannot
 * interleave differently.
 *
 * This order is what makes the async executor's one shared frame window
 * fair (SP2 R2): the executors admit frames in the order they are
 * submitted, so with k streams still reading, any M consecutive admissions
 * hold at most ceil(M/k) frames of one stream.
 */
class StreamReader {
 public:
    StreamReader(std::vector<StreamInput> inputs, std::size_t frames_per_stream);
    /// The next frame. *stream is its StageGraph::streams() index (inputs
    /// come from ResolveStreams, in the same order), *index its number in its
    /// stream. False once every stream has ended.
    bool Next(std::size_t* stream, cv::Mat* frame, std::size_t* index);
    std::size_t size() const;
    const StreamInput& input(std::size_t stream) const;
    std::size_t frames_read(std::size_t stream) const;
    bool ended(std::size_t stream) const;
    /// --loop: a video file that ends is opened again (OpenInput on its uri)
    /// and read from its first frame, so the stream runs until --frames or a
    /// stop request. Frame indices keep counting across restarts. Images,
    /// cameras and RTSP are unaffected; a video that gives no frame right
    /// after a restart ends there. Off by default (dx_graph never sets it).
    void SetLoop(bool loop);
    /// How many times stream `stream` started again.
    std::size_t restarts(std::size_t stream) const;

 private:
    /// Opens a video stream again from its start. False for anything that
    /// is not a video file, or when it will not open.
    bool Restart(std::size_t stream);

    std::vector<StreamInput> inputs_;
    std::vector<std::size_t> read_;
    std::vector<char> ended_;
    std::size_t limit_;
    std::size_t turn_;  ///< the stream whose turn it is
    bool loop_;
    std::vector<std::size_t> restarts_;
    std::vector<std::size_t> read_since_start_;  ///< frames since it (re)opened
};

// ---------------------------------------------------------------------
// The T1 resource stage: which .dxnn files a graph needs and lacks
// ---------------------------------------------------------------------

struct MissingArtifact {
    std::string node;
    std::string model;
    std::string path;
    std::string download_name;
    std::string variant;     ///< the registry key the node resolved to
    bool published;          ///< false: the model zoo has no file for it yet

    MissingArtifact() : published(true) {}
};

std::vector<MissingArtifact> CollectMissing(const GraphSpec& spec,
                                            const IModelRegistry& registry,
                                            const std::string& model_dir);

/// One line per absent artifact, one line per unpublished model (which
/// the model zoo cannot serve), then the one command that fetches the rest
/// into model_dir (DownloadCommand). Each caller prints its own header line
/// first, because --check reports this and a real run fails on it.
std::string FormatMissingList(const std::vector<MissingArtifact>& missing,
                              const std::string& model_dir);

/// "python3 scripts/download_models.py --models <names...> --output <dir>",
/// the recovery command for absent or broken .dxnn files. Not
/// "./setup.sh --models": setup.sh also re-fetches the sample videos, and
/// under its default --force scripts/get_resource.sh rm -rf's an
/// assets/videos symlink. download_models.py takes the manifest name, the
/// .dxnn name or the registry name, and overwrites by default. A name or
/// directory with characters a shell would split or expand is single-quoted.
std::string DownloadCommand(const std::vector<std::string>& names,
                            const std::string& model_dir);

/// "2 model files are not in <model_dir>".
std::string MissingHeadline(std::size_t count, const std::string& model_dir);

/// A model's declared extra output ports as "name=shape" joined by single
/// spaces, in declaration order: "drivable=labelmap lane=labelmap". ""
/// for a model without ports.
std::string PortsText(const ModelInfo& info);

/**
 * @brief A graph names .dxnn files that are not in model_dir.
 *
 * Not a GraphError, because the CLI never reported it as one:
 * StageGraph::Build would raise kModelMissing one model at a time, and
 * listing every absent file at once matters when one graph names three to
 * five models. what() is EXACTLY the text the CLI prints on stderr for
 * this case - the "ERROR [MODEL_MISSING] graph ..." line, the list, and the
 * trailing newline - so a consumer can print it unchanged.
 */
class MissingModelsError : public std::runtime_error {
 public:
    explicit MissingModelsError(const std::string& text)
        : std::runtime_error(text) {}
};

/// `"<name>" is the old name of "<variant>"`, or an alias of it (R6).
/// Empty when `name` is the model's own key.
std::string AliasNote(const std::string& name, const ModelInfo& info,
                      const std::map<std::string, ModelAlias>& aliases);

/// One `note: node "<id>": <AliasNote>` line per model node named by an old
/// name or an alias, in node order. --check prints these on stdout and a
/// run (PrepareGraph) on stderr, so a name never resolves silently.
std::string AliasNotes(const GraphSpec& spec, const IModelRegistry& registry);

/**
 * @brief The CLI's exact build sequence.
 *
 * Parse the file (T0), validate it against the registry (T0), print the
 * AliasNotes on stderr, check every .dxnn exists in model_dir (T1, fatal
 * here), build. Parse and validation
 * failures throw GraphError; absent artifacts throw MissingModelsError; a
 * failure inside Build throws whatever Build throws. *spec is assigned as
 * soon as the file parses; *graph is built only when every check passed.
 */
void PrepareGraph(const std::string& graph_path, const std::string& model_dir,
                  const IModelRegistry& registry, GraphSpec* spec,
                  StageGraph* graph);

// ---------------------------------------------------------------------
// Executors
// ---------------------------------------------------------------------

/// One shape for both executors, so an input loop is written once.
/// `stream` is an index into StageGraph::streams(); `index` the frame's
/// number in that stream.
class FrameExecutor {
 public:
    virtual ~FrameExecutor() {}
    virtual void Submit(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                        std::size_t index) = 0;
    virtual bool TryNext(FrameReport* report, cv::Mat* frame) = 0;
    virtual void Finish(StageGraph& graph) = 0;
};

class SyncFrameExecutor : public FrameExecutor {
 public:
    void Submit(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                std::size_t index) {
        Done done;
        done.report = inner_.RunFrame(graph, stream, frame, index);
        done.frame = frame.clone();
        ready_.push_back(done);
    }
    bool TryNext(FrameReport* report, cv::Mat* frame) {
        if (ready_.empty()) return false;
        *report = ready_.front().report;
        if (frame != NULL) *frame = ready_.front().frame;
        ready_.pop_front();
        return true;
    }
    void Finish(StageGraph&) {}

 private:
    struct Done {
        FrameReport report;
        cv::Mat frame;
    };
    SyncExecutor inner_;
    std::deque<Done> ready_;
};

class AsyncFrameExecutor : public FrameExecutor {
 public:
    explicit AsyncFrameExecutor(const AsyncOptions& options) : inner_(options) {}
    void Submit(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                std::size_t index) {
        inner_.Submit(graph, stream, frame, index);
    }
    bool TryNext(FrameReport* report, cv::Mat* frame) {
        return inner_.TryNext(report, frame);
    }
    void Finish(StageGraph& graph) { inner_.Finish(graph); }

 private:
    AsyncExecutor inner_;
};

/// What AsyncOptions::on_stuck reports, without the trailing newline:
/// "interrupted: still waiting on stage(s): a, b".
std::string StuckMessage(const std::vector<std::string>& ids);

}  // namespace consumer
}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_MULTI_MODEL_GRAPH_GRAPH_CONSUMER_HPP
