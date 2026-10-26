/**
 * @file stage_graph.hpp
 * @brief A validated GraphSpec plus live stages, ready to execute.
 *
 * Everything here is shared by both executors; the only difference between
 * sync and async is when work is submitted.
 */
#ifndef DXAPP_GRAPH_STAGE_GRAPH_HPP
#define DXAPP_GRAPH_STAGE_GRAPH_HPP

#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/graph/graph_config.hpp"
#include "common/graph/i_registry.hpp"
#include "common/graph/logic_ops.hpp"
#include "common/graph/roi_router.hpp"

namespace dxapp {
namespace graph {

struct NodeRuntime {
    std::string id;
    bool is_source;
    bool is_cpu;
    bool is_logic;
    std::string uri;
    std::string model_name;
    std::string op;                  ///< cpu only
    StageParams params;              ///< cpu only
    std::unique_ptr<IStage> stage;   ///< null for a source and a cpu node
    Shape output_shape;
    bool tracked;
    /// Per stream (StageGraph::streams() order), for a tracked node: its own
    /// IouTracker in every stream that runs it, null in the others. Empty for
    /// an untracked node. Track ids never cross streams.
    std::vector<std::shared_ptr<IouTracker> > trackers;
    /// Logic only: its params, typed (MakeLogicConfig), shared by every stream.
    std::shared_ptr<const LogicConfig> logic;
    /// Per stream (streams() order), for a logic node: its own state in every
    /// stream that runs it, null in the others. Never shared across streams.
    std::vector<std::shared_ptr<LogicState> > logic_states;
    /// Tracked: the "track" it was built with (a restarted stream's fresh tracker).
    TrackSpec track;
    /// Per stream (streams() order), for a tracked or logic node: the
    /// StreamPlan::generation its tracker and state belong to.
    std::vector<std::size_t> state_generation;

    NodeRuntime()
        : is_source(false), is_cpu(false), is_logic(false), output_shape(Shape::kFrame),
          tracked(false) {}
};

struct EdgeRuntime {
    std::size_t from;
    std::size_t to;
    RoiSpec roi;
    std::string port;  ///< "" = primary
    bool carry_result;

    EdgeRuntime() : from(0), to(0), carry_result(false) {}
};

/// What one edge hands on (U-08): the producer's primary payload, or the
/// payload of the port the edge names.
struct EdgeCargo {
    StageDataPtr payload;    ///< null: nothing to hand on
    const BoxesData* boxes;  ///< the payload's boxes, NULL when it has none
    EdgeCargo() : boxes(NULL) {}
};

/// The cargo of `edge`: `produced` and its (possibly tracked) `boxes` when
/// the edge reads the primary output, else the named entry of `ports` and
/// its box view. Both executors call it for every outgoing edge.
EdgeCargo SelectEdgeCargo(const EdgeRuntime& edge, const StageDataPtr& produced,
                          const BoxesData* boxes, const StagePorts& ports);

struct FrameReport {
    std::size_t frame_index;   ///< the caller's index, counted per stream
    std::string stream;        ///< the id of this frame's source node (every graph)
    std::map<std::string, StageResult> node_results;              ///< full-frame stages
    std::map<std::string, std::vector<StageResult> > roi_results; ///< ROI stages
    int skipped_out_of_bounds;
    std::string error;
    /// Milliseconds each non-source node took in this frame. Sync: inside the
    /// node's stage calls (every crop of a ROI node, summed) or its cpu
    /// reduction. Async: from the node's dispatch to its last job's recorded
    /// completion, waiting behind other frames included; 0 for a ROI node
    /// with no crops. Measured, so NOT compared by operator== and written to
    /// --report only with --report-timing.
    std::map<std::string, double> node_ms;
    /// Seconds: StageGraph::FrameTime of this frame, the clock logic nodes
    /// count in. Set once per frame, so every logic node sees one time.
    double time_s;
    /// Every event the logic nodes raised in this frame, in the stream's
    /// topological node order (each node's events in the order it raised them).
    std::vector<GraphEvent> events;
    /// What a logic node asks the visualizer to draw (pass 5), by node id;
    /// only a node with something to draw this frame has an entry.
    std::map<std::string, LogicOverlay> overlays;
    /// Every logic node that ran in this frame. The visualizer's shape
    /// passes skip their payloads: a filter's or a zone's boxes are its
    /// detector's boxes, already drawn.
    std::set<std::string> logic_nodes;

    FrameReport() : frame_index(0), skipped_out_of_bounds(0), time_s(0.0) {}
};

/// Milliseconds from `since` to `until`, for FrameReport::node_ms; never negative.
inline double ElapsedMs(std::chrono::steady_clock::time_point since,
                        std::chrono::steady_clock::time_point until) {
    const double ms = std::chrono::duration<double, std::milli>(until - since).count();
    return ms > 0.0 ? ms : 0.0;
}

/// Structural equality, used by the sync/async parity test.
bool operator==(const FrameReport& a, const FrameReport& b);
inline bool operator!=(const FrameReport& a, const FrameReport& b) {
    return !(a == b);
}

/**
 * @brief Canonical order for a node's ROI results.
 *
 * Orders by (parent_node, parent_index, roi_index), ascending. A node's ROI
 * results can arrive in whatever order its producer(s) emitted them in
 * (RouteRois sorts by score for the max-cut step, not by index; a node with
 * two ROI parents interleaves both producers' crops in producer-processing
 * order) — this is the single comparator both executors sort into before
 * storing roi_results, so Task 10's parity test compares like-ordered
 * vectors regardless of either executor's internal delivery order. Declared
 * here, not duplicated per executor, so sync and async cannot drift apart
 * on what "canonical order" means.
 *
 * Tie safety (review round 3, item 1): a full three-way tie — same
 * parent_node AND same parent_index AND same roi_index — would leave
 * std::stable_sort's output order dependent on delivery order, which is
 * exactly what a parity comparator must not depend on. That tie cannot
 * occur for any node in a validated graph: matching on parent_node alone
 * already requires two crops in one node's inbox to have arrived via two
 * edges sharing the same "from", and combined with the shared "to" (the
 * consumer whose inbox they share), that is exactly the duplicate
 * (from, to) edge pair ValidateGraph rejects unconditionally, regardless of
 * each edge's "roi" contents (graph_config.cpp:417-424, thrown as
 * kGraphSchema "duplicate edge"). Pinned by the existing
 * TestValidateRejectsDuplicateEdge (graph_engine_test.cpp:906-920), whose
 * fixture is precisely two ROI edges from one parent into one consumer —
 * the exact shape that would otherwise produce this tie. Build() is
 * documented to run only on a validated GraphSpec (see its own kModelUnknown
 * guard's comment), so ByOrigin can rely on this rejection without adding a
 * source-edge discriminator to RoiRef.
 */
bool ByOrigin(const StageResult& a, const StageResult& b);

/// Frames per second of a stream whose source reports none (an image,
/// a camera that does not say) and of every stream until SetStreamClock.
const double kDefaultStreamFps = 30.0;

/// One stream of a built graph (SP2): the frames of one source node.
struct StreamPlan {
    std::size_t source;              ///< node index of the source
    std::string source_id;           ///< its id: FrameReport::stream of this stream
    std::vector<char> member;        ///< per node: 1 when this stream runs it
    std::vector<int> in_edges;       ///< per node: edges into it whose producer is a member
    std::vector<std::size_t> order;  ///< topological_order(), members only
    double fps;                      ///< frame_index / fps is a file frame's time
    bool wall_clock;                 ///< a live source: seconds since its first frame
    bool clock_started;
    std::chrono::steady_clock::time_point clock_start;
    /// How often the stream started again (RestartStream): a frame admitted
    /// in a later generation finds fresh trackers and logic state.
    std::size_t generation;
    StreamPlan()
        : source(0), fps(kDefaultStreamFps), wall_clock(false), clock_started(false),
          generation(0) {}
};

class StageGraph {
 public:
    /// Throws GraphError (kModelMissing, kModelLoad): kModelMissing when
    /// require_artifacts is true and a node's resolved .dxnn file is not on
    /// disk; kModelLoad when IModelRegistry::createStage throws for a node
    /// (the runtime could not load its engine), naming the node and keeping
    /// the runtime's text. Tests that build against a scripted,
    /// hardware-free registry with no real artifacts on disk pass
    /// require_artifacts=false to stay hermetic; the production CLI path
    /// leaves it at its default (true).
    void Build(const GraphSpec& spec, const IModelRegistry& registry,
               const std::string& model_dir, bool require_artifacts = true);

    const std::vector<NodeRuntime>& nodes() const { return nodes_; }
    std::vector<NodeRuntime>& mutable_nodes() { return nodes_; }
    const std::vector<EdgeRuntime>& edges() const { return edges_; }
    const std::vector<std::size_t>& topological_order() const { return order_; }
    const std::vector<std::size_t>& source_indices() const { return sources_; }

    /// One per source node, in declaration order (streams()[k].source == source_indices()[k]).
    const std::vector<StreamPlan>& streams() const { return streams_; }
    /// The index into streams() of source node `source_id`; streams().size() if none.
    std::size_t StreamIndex(const std::string& source_id) const;
    /// 0 for a one-source graph. Otherwise throws std::invalid_argument:
    /// "<caller>: the graph has 2 source nodes (camA, camB), so each frame must
    /// name its stream (an index into StageGraph::streams())", or
    /// "<caller>: the graph has no source node".
    std::size_t OnlyStream(const char* caller) const;
    /// How stream `stream` counts time (logic nodes' seconds): frame_index /
    /// fps, or with `wall_clock` (a camera, RTSP) the seconds since the first
    /// frame asked for after this call. A rate that is not a positive finite
    /// number keeps kDefaultStreamFps. Build resets every stream to
    /// frame_index / kDefaultStreamFps. Throws std::out_of_range.
    void SetStreamClock(std::size_t stream, double fps, bool wall_clock);
    /// The time of frame `frame_index` of stream `stream`, in seconds. Both
    /// executors call it once per frame, from the driver thread.
    double FrameTime(std::size_t stream, std::size_t frame_index);
    /// Stream `stream` starts again from its first frame (--loop restarts a
    /// video): every tracker and logic node of the stream starts afresh at
    /// the next frame admitted - new track ids, no zone members, line
    /// history, stays, counts or condition edge - as on a first pass. Frame
    /// indices and stream time keep counting. Call it between frames, from
    /// the driver thread. Throws std::out_of_range.
    void RestartStream(std::size_t stream);
    /// Before node `index` uses its tracker or logic state for a frame of
    /// stream `stream` admitted in `generation` (StreamPlan::generation at
    /// its admission): a fresh one when the stream restarted since the
    /// node's last frame of it. The executors call it in the stream's frame
    /// order, on the driver thread.
    void FreshenStreamState(std::size_t index, std::size_t stream, std::size_t generation);

    /// Indices of edges whose producer is `node`.
    std::vector<std::size_t> OutgoingEdges(std::size_t node) const;

 private:
    std::vector<NodeRuntime> nodes_;
    std::vector<EdgeRuntime> edges_;
    std::vector<std::size_t> order_;
    std::vector<std::size_t> sources_;
    std::vector<StreamPlan> streams_;
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_STAGE_GRAPH_HPP
