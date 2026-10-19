#include "common/graph/graph_runner_async.hpp"

#include "common/graph/cpu_reduce.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/graph/roi_router.hpp"

namespace dxapp {
namespace graph {
namespace {

/// AsyncOptions::stall_timeout_ms as a duration that cannot overflow. The
/// steady clock counts nanoseconds in 64 bits, so milliseconds(ms) compared
/// against it is multiplied by 10^6: past about 9.2e12 ms that wraps
/// negative, and a size_t above INT64_MAX is negative already - either way
/// "wait N ms" became "stall at once". Anything above the cap (about 31
/// years) is the same as waiting forever, and dx_graph passes any
/// non-negative Python int through, so the executor clamps rather than the
/// callers.
std::chrono::milliseconds StallTimeout(std::size_t ms) {
    const std::size_t kCapMs = static_cast<std::size_t>(1000000000000ULL);
    return std::chrono::milliseconds(
        static_cast<std::chrono::milliseconds::rep>(ms < kCapMs ? ms : kCapMs));
}

/**
 * @brief Everything one node's in-flight jobs write into.
 *
 * Held by shared_ptr and captured BY VALUE in every completion callback, not
 * by reference to a stack local. Two reasons, both from the hardware notes in
 * i_registry.hpp: a real runtime fires callbacks on its own thread, so a
 * completion that arrives after the dispatching scope has exited must not
 * write through a dangling reference; and the data a callback is handed
 * dangles once it returns, so whatever is needed is copied out here (a
 * StageResult copy holds its own RoiRef and a counted reference to the
 * payload) before the callback finishes.
 *
 * SYNCHRONIZED, and not as a formality. Every write below happens on
 * whatever thread the runtime fires the completion on (IStage's THREADING
 * contract in i_registry.hpp: callbacks are NOT guaranteed to be
 * serialized), while the reads happen on the driver thread when the node
 * is harvested.
 * Record() is a read-modify-write of delivered_ followed by a push_back
 * that can REALLOCATE results_; two concurrent completions doing that
 * unsynchronized is undefined behaviour whose failure mode is memory
 * corruption, not a clean test failure. Before this, the code relied on an
 * undocumented property of dxrt - dxrt_cxx_api.h:563-578 promises nothing
 * about which thread, or how many at once - so the mutex is here rather
 * than the assumption.
 *
 * delivered_[] is also the double-delivery guard. A blocking Run() fires
 * the registered callback too (Task 3 spike: 5 Run() calls, zero RunAsync,
 * exactly 5 callback fires), so one logical job can be announced twice;
 * appending blindly would report a crop twice and silently change the ROI
 * count. Claiming and writing under ONE lock is what makes that guard
 * atomic rather than merely present.
 *
 * errors_[] is indexed by job, not appended in completion order, so the
 * surviving message is chosen by job index afterwards. SyncExecutor's loop
 * overwrites per crop, so the last crop's message wins; picking the
 * last non-empty entry here reproduces that regardless of arrival order.
 */
class JobCollector {
 public:
    explicit JobCollector(std::size_t jobs)
        : errors_(jobs), delivered_(jobs, 0) {}

    /// Claim a job and record its outcome, atomically. Returns false for a
    /// second fire of the same job or an out-of-range index, so the caller
    /// announces each job exactly once.
    bool Record(std::size_t job, const StageResult& result,
                const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (job >= delivered_.size() || delivered_[job] != 0) return false;
        delivered_[job] = 1;
        last_done_ = std::chrono::steady_clock::now();
        if (!error.empty()) {
            errors_[job] = error;
            return true;
        }
        results_.push_back(result);  // copy before it dangles
        return true;
    }

    std::vector<StageResult> Results() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_;
    }

    std::string LastError() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string last;
        for (std::size_t i = 0; i < errors_.size(); ++i) {
            if (!errors_[i].empty()) last = errors_[i];
        }
        return last;
    }

    /// When the last accepted completion was recorded, on the thread that
    /// delivered it (the epoch when none was).
    std::chrono::steady_clock::time_point LastDone() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_done_;
    }

 private:
    mutable std::mutex mutex_;
    std::vector<StageResult> results_;
    std::vector<std::string> errors_;
    std::vector<char> delivered_;
    std::chrono::steady_clock::time_point last_done_;
};
typedef std::shared_ptr<JobCollector> JobCollectorPtr;

/// "One job of node `node` in frame `seq` came back."
struct Notice {
    std::size_t seq;
    std::size_t node;
};

/**
 * @brief Where callbacks announce completions.
 *
 * Held by shared_ptr and captured by every callback, so a completion that
 * arrives after the executor is gone writes into memory that still exists
 * rather than through a dangling pointer. The lock is also what orders the
 * callback thread's writes before the driver's reads; nothing here relies on
 * flush()'s happens-before edge.
 */
class CompletionQueue {
 public:
    void Push(const Notice& notice) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            items_.push_back(notice);
        }
        ready_.notify_one();
    }

    std::vector<Notice> TakeAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Notice> out(items_.begin(), items_.end());
        items_.clear();
        return out;
    }

    void WaitFor(std::size_t ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_for(lock, std::chrono::milliseconds(ms),
                        [this] { return !items_.empty(); });
    }

 private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Notice> items_;
};
typedef std::shared_ptr<CompletionQueue> CompletionQueuePtr;

/// One dispatched node in one frame, until it is harvested.
struct NodeRun {
    JobCollectorPtr jobs;
    bool is_roi;
    bool is_logic;  ///< a logic node: no jobs, it runs at its gate turn (Harvest)
    std::size_t total;      ///< jobs this run owns
    std::size_t completed;  ///< accepted completions processed so far
    std::chrono::steady_clock::time_point started;  ///< when Dispatch ran (node_ms)
    NodeRun() : is_roi(false), is_logic(false), total(0), completed(0) {}
};

/// Everything one frame needs while it is in flight (spec 4.5 FrameContext).
struct FrameCtx {
    std::size_t seq;
    std::size_t stream;  ///< StageGraph::streams() index
    std::size_t stream_seq;  ///< admission order within its stream: the key of its tracker gates
    std::size_t generation;  ///< StreamPlan::generation at admission (RestartStream)
    FrameReport report;
    cv::Mat frame;  ///< an owned copy: sources reuse their buffer
    std::vector<int> pending_inputs;
    std::vector<std::vector<RoiCrop> > inbox;
    std::vector<std::vector<StageResult> > result_in;  ///< carry-result payloads and origins, per node
    std::vector<std::vector<LogicInput> > logic_in;  ///< what a logic node's edges brought
    std::map<std::size_t, std::vector<GraphEvent> > node_events;  ///< per logic node
    std::vector<char> got_frame;
    /// The view a released plain edge delivered; valid where got_frame is 1.
    std::vector<FrameView> frame_in;
    /// The view a full-frame node ran on (the source's is {frame, identity}).
    std::vector<FrameView> view;
    std::vector<char> done;  ///< dispatched (or resolved, for a source)
    std::map<std::size_t, std::string> node_errors;
    std::map<std::size_t, NodeRun> runs;  ///< dispatched, not yet harvested
    std::vector<std::size_t> ready;       ///< released, not yet dispatched
    bool closed;
    FrameCtx() : seq(0), stream(0), stream_seq(0), generation(0), closed(false) {}
};
typedef std::shared_ptr<FrameCtx> FrameCtxPtr;

/// A job waiting for a slot on its stage (max_jobs_per_stage).
struct QueuedJob {
    FrameCtxPtr frame;
    /// The run's collector, carried rather than looked up in frame->runs at
    /// submit time: a job can wait in a backlog behind other frames' jobs,
    /// and a lookup by operator[] would silently insert an empty NodeRun if
    /// the run were ever gone by then.
    JobCollectorPtr jobs;
    std::size_t job;
    StageInput input;
};

/// One (stream, tracked node) ordering gate (parity rule 2), keyed by
/// FrameCtx::stream_seq.
struct TrackGate {
    std::size_t next_seq;                     ///< the only stream_seq that may track now
    std::map<std::size_t, FrameCtxPtr> held;  ///< completed, waiting for their turn
    std::set<std::size_t> skips;              ///< stream_seqs that will never arrive
    TrackGate() : next_seq(0) {}
};

struct Finished {
    FrameReport report;
    cv::Mat frame;
};

/**
 * @brief Decrement every successor's counter, and hand on a payload if there
 *        is one.
 *
 * Releasing an edge and sending data along it are separate steps, and they
 * have to be. SyncExecutor walks every node in topological order whatever
 * its parents did, so a consumer whose producer failed, produced nothing, or
 * was itself a ROI stage still runs and still lands in the report with an
 * empty result. Here a consumer only runs if something releases it, so the
 * counter is decremented unconditionally while the payload is propagated
 * only when there is one. Decrementing only on success is the hang in Review
 * Focus #1; propagating unconditionally invents a full frame for a consumer
 * that sync leaves empty.
 *
 * Releasing still decrements unconditionally. What a plain edge delivers
 * when there is a payload: from a source, the source's view; from any other
 * producer, HandOffView over the producer's image and the view it ran on -
 * or, when HandOffProblem rejects that image, nothing, and the consumer's
 * hand-off error is recorded in node_errors instead (it then runs as an
 * empty ROI consumer, exactly as sync's no-input branch does). ROI edges
 * route over the producer's view; one whose port hands off no boxes records
 * the consumer's error the same way. What an edge carries is SelectEdgeCargo's
 * choice: `produced` (and `boxes`), or the payload of the port it names.
 *
 * `crops`: a per-crop node's results (ByOrigin order), handed to each logic
 * node on a "carry": "result" edge; NULL for every other producer.
 *
 * Shared by the source path and the harvest path so the two cannot drift.
 */
void ReleaseSuccessors(const StageGraph& graph, std::size_t index,
                       const StageDataPtr& produced, const BoxesData* boxes,
                       const StagePorts& ports, const std::vector<StageResult>* crops,
                       FrameCtx& ctx) {
    const std::vector<EdgeRuntime>& edges = graph.edges();
    const std::vector<std::size_t> outgoing = graph.OutgoingEdges(index);
    const std::string& producer_id = graph.nodes()[index].id;

    for (std::size_t e = 0; e < outgoing.size(); ++e) {
        const EdgeRuntime& edge = edges[outgoing[e]];
        if (crops != NULL && edge.carry_result) {
            if (graph.nodes()[edge.to].is_logic) {
                LogicInput input;
                input.edge = outgoing[e];
                input.from = producer_id;
                input.per_crop = true;
                input.crops = *crops;
                ctx.logic_in[edge.to].push_back(input);
            }
        } else if (produced) {
            const EdgeCargo cargo = SelectEdgeCargo(edge, produced, boxes, ports);
            if (edge.carry_result) {
                if (cargo.payload && graph.nodes()[edge.to].is_logic) {
                    LogicInput input;
                    input.edge = outgoing[e];
                    input.from = producer_id;
                    input.data = cargo.payload;
                    std::map<std::string, StageResult>::const_iterator mine =
                        ctx.report.node_results.find(producer_id);
                    if (mine != ctx.report.node_results.end()) input.origin = mine->second.origin;
                    ctx.logic_in[edge.to].push_back(input);
                } else if (cargo.payload) {
                    StageResult input;
                    input.data = cargo.payload;
                    std::map<std::string, StageResult>::const_iterator mine =
                        ctx.report.node_results.find(producer_id);
                    if (mine != ctx.report.node_results.end()) input.origin = mine->second.origin;
                    ctx.result_in[edge.to].push_back(input);
                }
            } else if (!edge.roi.present) {
                if (graph.nodes()[index].is_source) {
                    ctx.frame_in[edge.to] = ctx.view[index];
                    ctx.got_frame[edge.to] = 1;
                } else {
                    const std::string problem = HandOffProblem(cargo.payload);
                    if (problem.empty()) {
                        ctx.frame_in[edge.to] = HandOffView(
                            *static_cast<const ImageData*>(cargo.payload.get()), ctx.view[index]);
                        ctx.got_frame[edge.to] = 1;
                    } else {
                        ctx.node_errors[edge.to] = "node \"" + graph.nodes()[edge.to].id +
                                                   "\": \"" + producer_id + "\" " + problem;
                    }
                }
            } else if (cargo.boxes != NULL) {
                RouteStats stats;
                const std::vector<RoiCrop> crops = RouteRois(
                    *cargo.boxes, ctx.view[index], edge.roi, producer_id, NULL, &stats);
                ctx.report.skipped_out_of_bounds += stats.clipped_away;
                // Append, never assign: a node can have more than one ROI
                // parent, and each parent is released separately. Assigning
                // silently drops every earlier parent's crops - the defect
                // Task 9's review round 1 found in the synchronous executor.
                std::vector<RoiCrop>& target = ctx.inbox[edge.to];
                target.insert(target.end(), crops.begin(), crops.end());
            } else if (!edge.port.empty()) {
                // A port that is missing or not box-shaped breaks the
                // producer's ports contract: recorded as SyncExecutor does.
                ctx.node_errors[edge.to] = "node \"" + graph.nodes()[edge.to].id +
                                           "\": \"" + producer_id + "\" port \"" +
                                           edge.port + "\" handed off no boxes";
            }
        }
        if (--ctx.pending_inputs[edge.to] == 0) ctx.ready.push_back(edge.to);
    }
}

std::string JoinIds(const std::vector<std::string>& ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += ", ";
        out += "\"" + ids[i] + "\"";
    }
    return out;
}

}  // namespace

struct AsyncExecutor::Engine {
    AsyncOptions options;
    CompletionQueuePtr queue;
    StageGraph* graph;
    std::size_t next_seq;
    std::size_t next_emit;
    std::size_t tick;
    std::map<std::size_t, FrameCtxPtr> frames;  ///< admitted, not finished
    std::map<std::size_t, Finished> finished;   ///< finished, not handed out
    std::vector<std::size_t> outstanding;       ///< per node: submitted, not processed
    std::vector<std::deque<QueuedJob> > backlog;       ///< per node: waiting for a slot
    /// Per node, one tick per outstanding job. A completion pops the FRONT,
    /// whichever job it was, so "oldest outstanding submission" is
    /// approximate; it only orders poll() calls, which are no-ops on hardware.
    std::vector<std::deque<std::size_t> > submit_ticks;
    /// Per node, not per frame: an escaped poll() lands in the next harvest
    /// of that node, reachable only through a contract-violating stage.
    std::vector<std::string> stage_failure;
    /// Per stream, then per tracked node that stream runs (parity rule 2,
    /// SP2): a gate orders ONE stream's frames by stream_seq. A frame of
    /// another stream never reaches it, holds it or moves it.
    std::vector<std::map<std::size_t, TrackGate> > gates;
    /// Per stream: the stream_seq the next admitted frame of it gets.
    std::vector<std::size_t> next_stream_seq;
    std::size_t peak_in_flight;
    bool stuck_reported;

    explicit Engine(const AsyncOptions& opts)
        : options(opts), queue(new CompletionQueue()), graph(NULL),
          next_seq(0), next_emit(0), tick(0), peak_in_flight(0),
          stuck_reported(false) {}

    void Bind(StageGraph& target) {
        // Neither the address nor the node count is identity:
        // StageGraph::Build can be called again on the same object - with the
        // same node count but a different tracked set - and a new graph can
        // reuse an old address. So whenever nothing is in flight, re-derive
        // every per-graph structure from `target`; it costs one pass over the
        // nodes per idle Submit. Untaken finished reports do not make the
        // executor busy: they hold no per-graph state.
        const std::size_t n = target.nodes().size();
        if (!frames.empty()) {
            if (graph == &target && outstanding.size() == n &&
                backlog.size() == n && submit_ticks.size() == n &&
                stage_failure.size() == n && gates.size() == target.streams().size()) {
                return;
            }
            throw std::logic_error(
                "AsyncExecutor: a different StageGraph was passed while frames "
                "are still in flight");
        }
        graph = &target;
        outstanding.assign(n, 0);
        backlog.assign(n, std::deque<QueuedJob>());
        submit_ticks.assign(n, std::deque<std::size_t>());
        stage_failure.assign(n, std::string());
        const std::vector<NodeRuntime>& nodes = target.nodes();
        const std::size_t streams = target.streams().size();
        if (next_stream_seq.size() != streams) next_stream_seq.assign(streams, 0);
        gates.assign(streams, std::map<std::size_t, TrackGate>());
        for (std::size_t s = 0; s < streams; ++s) {
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                if ((nodes[i].tracked || nodes[i].is_logic) && target.streams()[s].member[i]) {
                    gates[s][i].next_seq = next_stream_seq[s];
                }
            }
        }
    }

    FrameCtxPtr Admit(const cv::Mat& frame, std::size_t stream, std::size_t frame_index) {
        const StreamPlan& plan = graph->streams()[stream];
        FrameCtxPtr ctx(new FrameCtx());
        ctx->seq = next_seq++;
        ctx->stream = stream;
        ctx->stream_seq = next_stream_seq[stream]++;
        ctx->report.frame_index = frame_index;
        ctx->report.stream = plan.source_id;
        ctx->report.time_s = graph->FrameTime(stream, frame_index);
        ctx->generation = plan.generation;
        if (frame.empty()) {
            ctx->report.error = "source produced an empty frame";
            ctx->closed = true;
            Finished done;
            done.report = ctx->report;
            finished[ctx->seq] = done;
            // Only this stream's gates: no other stream's frame waits for it.
            for (std::map<std::size_t, TrackGate>::iterator g = gates[stream].begin();
                 g != gates[stream].end(); ++g) {
                SkipGate(stream, g->first, ctx->stream_seq);
            }
            return ctx;
        }
        ctx->frame = frame.clone();
        const std::size_t n = graph->nodes().size();
        // The stream's in-edges only: a node two sources feed is released by
        // the one edge this stream brings it.
        ctx->pending_inputs = plan.in_edges;
        ctx->inbox.assign(n, std::vector<RoiCrop>());
        ctx->result_in.assign(n, std::vector<StageResult>());
        ctx->logic_in.assign(n, std::vector<LogicInput>());
        ctx->got_frame.assign(n, 0);
        ctx->frame_in.assign(n, FrameView());
        ctx->view.assign(n, FrameView());
        ctx->done.assign(n, 0);
        ctx->ready.assign(1, plan.source);
        frames[ctx->seq] = ctx;
        if (frames.size() > peak_in_flight) peak_in_flight = frames.size();
        DispatchReady(ctx);
        FinishIfQuiet(ctx);
        return ctx;
    }

    /// True when this frame already holds an error from before node `index`
    /// in its stream's topological order: a cpu or logic node's own error
    /// then does not replace it, as in SyncExecutor.
    bool HasEarlierError(const FrameCtx& ctx, std::size_t index) const {
        if (!ctx.report.error.empty()) return true;
        const std::vector<std::size_t>& order = graph->streams()[ctx.stream].order;
        for (std::size_t o = 0; o < order.size() && order[o] != index; ++o) {
            if (ctx.node_errors.count(order[o]) != 0) return true;
        }
        return false;
    }

    void DispatchReady(const FrameCtxPtr& ctx) {
        std::vector<NodeRuntime>& nodes = graph->mutable_nodes();
        while (!ctx->ready.empty()) {
            const std::size_t index = ctx->ready.front();
            ctx->ready.erase(ctx->ready.begin());
            if (ctx->done[index]) continue;
            ctx->done[index] = 1;
            NodeRuntime& node = nodes[index];
            if (node.is_source) {
                std::shared_ptr<FrameData> data(new FrameData());
                data->image = ctx->frame;
                StageResult result;
                result.data = data;
                ctx->report.node_results[node.id] = result;
                FrameView source;
                source.image = ctx->frame;
                ctx->view[index] = source;
                ReleaseSuccessors(*graph, index, StageDataPtr(data), NULL, StagePorts(), NULL,
                                  *ctx);
                continue;
            }
            if (node.is_cpu) {
                const std::chrono::steady_clock::time_point started =
                    std::chrono::steady_clock::now();
                const CpuReduction reduced =
                    ReduceCpu(node.op, node.params, ctx->result_in[index], ctx->frame.size());
                ctx->report.node_ms[node.id] = ElapsedMs(started, std::chrono::steady_clock::now());
                ctx->report.node_results[node.id] = reduced.result;
                if (!IsEmpty(reduced.overlay)) ctx->report.overlays[node.id] = reduced.overlay;
                if (!reduced.error.empty() && !HasEarlierError(*ctx, index)) {
                    ctx->node_errors[index] =
                        std::string("node \"") + node.id + "\": " + reduced.error;
                }
                FrameView source;
                source.image = ctx->frame;
                ctx->view[index] = source;
                ReleaseSuccessors(*graph, index, reduced.result.data, NULL, StagePorts(), NULL,
                                  *ctx);
                continue;
            }
            if (node.is_logic) {
                // Gated like a tracked node (Bind): its state must see this
                // stream's frames in order. CompleteRun runs it now or parks
                // the frame until its turn; the run keeps the frame open.
                NodeRun run;
                run.is_logic = true;
                run.started = std::chrono::steady_clock::now();
                ctx->runs[index] = run;
                CompleteRun(ctx, index);
                continue;
            }
            Dispatch(ctx, index);
        }
    }

    void Dispatch(const FrameCtxPtr& ctx, std::size_t index) {
        NodeRun run;
        run.started = std::chrono::steady_clock::now();
        if (ctx->got_frame[index]) {
            run.total = 1;
            run.jobs.reset(new JobCollector(1));
            ctx->runs[index] = run;
            QueuedJob job;
            job.frame = ctx;
            job.jobs = run.jobs;
            job.job = 0;
            const FrameView& in = ctx->frame_in[index];
            ctx->view[index] = in;
            job.input.image = in.image;
            job.input.origin.inv_align = in.to_source;
            Enqueue(index, job);
            return;
        }
        // ROI consumer: one job per crop. Zero crops is a legitimate
        // completion - a detector that found nothing still releases its
        // consumers, which finish empty rather than hang.
        const std::vector<RoiCrop>& crops = ctx->inbox[index];
        run.is_roi = true;
        run.total = crops.size();
        run.jobs.reset(new JobCollector(crops.size()));
        ctx->runs[index] = run;
        if (crops.empty()) {
            CompleteRun(ctx, index);
            return;
        }
        for (std::size_t c = 0; c < crops.size(); ++c) {
            QueuedJob job;
            job.frame = ctx;
            job.jobs = run.jobs;
            job.job = c;
            job.input.image = crops[c].image;
            job.input.origin = crops[c].ref;
            Enqueue(index, job);
        }
    }

    void Enqueue(std::size_t index, const QueuedJob& job) {
        backlog[index].push_back(job);
        SubmitBacklog(index);
    }

    /// Back-pressure: at most max_jobs_per_stage jobs outstanding on ONE
    /// stage, across frames. The rest wait here in submission order and go
    /// out as completions free slots. The batch boundary changes only
    /// arrival order, so it must not change the report
    /// (TestAsyncMatchesSyncUnderBackPressure).
    void SubmitBacklog(std::size_t index) {
        NodeRuntime& node = graph->mutable_nodes()[index];
        const std::size_t cap = options.max_jobs_per_stage;
        while (!backlog[index].empty() &&
               (cap == 0 || outstanding[index] < cap)) {
            const QueuedJob job = backlog[index].front();
            backlog[index].pop_front();
            ++outstanding[index];
            submit_ticks[index].push_back(tick++);
            SubmitJob(node, job, index);
        }
    }

    /// submit() with the guarantee SyncExecutor's try/catch gives run(): a
    /// throwing stage becomes this job's error, not a crash. The callback
    /// may run on any thread and may run inline; it only records and
    /// announces - all processing happens on the driver thread.
    void SubmitJob(NodeRuntime& node, const QueuedJob& job, std::size_t index) {
        const JobCollectorPtr jobs = job.jobs;
        const CompletionQueuePtr sink = queue;
        const Notice notice = {job.frame->seq, index};
        const std::size_t slot = job.job;
        try {
            node.stage->submit(job.input, [jobs, sink, notice, slot](
                                              const StageResult& result,
                                              const std::string& error) {
                if (jobs->Record(slot, result, error)) sink->Push(notice);
            });
        } catch (const std::exception& error) {
            if (jobs->Record(slot, StageResult(), error.what())) sink->Push(notice);
        } catch (...) {
            if (jobs->Record(slot, StageResult(), "unknown submit failure")) {
                sink->Push(notice);
            }
        }
    }

    /// Process every completion announced so far. Returns false if there
    /// were none.
    bool DrainQueue() {
        const std::vector<Notice> got = queue->TakeAll();
        if (got.empty()) return false;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const Notice& notice = got[i];
            --outstanding[notice.node];
            if (!submit_ticks[notice.node].empty()) {
                submit_ticks[notice.node].pop_front();
            }
            std::map<std::size_t, FrameCtxPtr>::iterator frame =
                frames.find(notice.seq);
            if (frame != frames.end()) {
                const FrameCtxPtr ctx = frame->second;
                std::map<std::size_t, NodeRun>::iterator run =
                    ctx->runs.find(notice.node);
                if (run != ctx->runs.end() &&
                    ++run->second.completed == run->second.total) {
                    CompleteRun(ctx, notice.node);
                }
            }
            SubmitBacklog(notice.node);
        }
        return true;
    }

    void CompleteRun(const FrameCtxPtr& ctx, std::size_t index) {
        std::map<std::size_t, TrackGate>& stream_gates = gates[ctx->stream];
        std::map<std::size_t, TrackGate>::iterator gate = stream_gates.find(index);
        if (gate != stream_gates.end() && ctx->stream_seq != gate->second.next_seq) {
            gate->second.held[ctx->stream_seq] = ctx;  // stays in ctx->runs until its turn
            return;
        }
        Harvest(*ctx, index);
        DispatchReady(ctx);
        FinishIfQuiet(ctx);
        if (gate != stream_gates.end()) {
            ++gate->second.next_seq;
            DrainGate(ctx->stream, index);
        }
    }

    /// The gate of tracked node `index` in stream `stream`. Bind creates one
    /// per (stream, tracked member node) and nothing else may: operator[]
    /// would silently add a gate for an untracked node, so a caller passing
    /// one is a scheduler defect.
    TrackGate& GateOf(std::size_t stream, std::size_t index) {
        std::map<std::size_t, TrackGate>& stream_gates = gates[stream];
        std::map<std::size_t, TrackGate>::iterator gate = stream_gates.find(index);
        if (gate == stream_gates.end()) {
            throw std::logic_error("AsyncExecutor: no tracker gate for node " +
                                   std::to_string(index) + " in stream " +
                                   std::to_string(stream));
        }
        return gate->second;
    }

    /// Release, in order, every parked or skipped frame of stream `stream`
    /// whose turn it now is at this per-stream gate.
    ///
    /// Re-entrant: the DispatchReady / FinishIfQuiet calls below (and in
    /// CompleteRun) can reach SkipGate or DrainGate again, for this per-stream
    /// gate or another. That is safe because the state is re-read on every
    /// iteration, and nothing erases from `gates` (only Bind does, and never
    /// while a frame is in flight), so the reference stays valid. Here
    /// next_seq advances before the nested calls; CompleteRun advances after
    /// them, which is also safe - while next_seq still names the frame just
    /// harvested, nothing nested can match it (that frame has passed this
    /// node), so later frames are only parked or marked skipped, and the
    /// DrainGate that follows releases them.
    void DrainGate(std::size_t stream, std::size_t index) {
        TrackGate& gate = GateOf(stream, index);
        for (;;) {
            if (gate.skips.erase(gate.next_seq) != 0) {
                ++gate.next_seq;
                continue;
            }
            std::map<std::size_t, FrameCtxPtr>::iterator held =
                gate.held.find(gate.next_seq);
            if (held == gate.held.end()) return;
            const FrameCtxPtr ctx = held->second;
            gate.held.erase(held);
            Harvest(*ctx, index);
            ++gate.next_seq;
            DispatchReady(ctx);
            FinishIfQuiet(ctx);
        }
    }

    /// A frame that will never reach this node must still move its stream's
    /// gate on, or every later frame of the stream waits for it forever.
    void SkipGate(std::size_t stream, std::size_t index, std::size_t stream_seq) {
        TrackGate& gate = GateOf(stream, index);
        if (stream_seq == gate.next_seq) {
            ++gate.next_seq;
            DrainGate(stream, index);
        } else {
            gate.skips.insert(stream_seq);
        }
    }

    void Harvest(FrameCtx& ctx, std::size_t index) {
        if (graph->nodes()[index].is_logic) {
            HarvestLogic(ctx, index);
            return;
        }
        NodeRuntime& node = graph->mutable_nodes()[index];
        const NodeRun run = ctx.runs[index];
        ctx.runs.erase(index);
        const std::string job_error = run.jobs->LastError();
        std::vector<StageResult> results = run.jobs->Results();
        // Dispatch to the last completion (spec: graph composer node time). A
        // ROI node that got no crops ran nothing.
        ctx.report.node_ms[node.id] =
            run.total == 0 ? 0.0 : ElapsedMs(run.started, run.jobs->LastDone());
        StageDataPtr produced;
        StagePorts ports;  // the producer's extra outputs (U-08)

        // The first error recorded for a node in this frame is kept: a
        // consumer whose hand-off error was recorded at release keeps it
        // through its zero-crop run, as SyncExecutor reports it. An escaped
        // poll() is taken only when it is recorded; otherwise it stays with
        // the node for its next harvest instead of being dropped.
        if (ctx.node_errors.count(index) == 0) {
            std::string stage_error;
            stage_error.swap(stage_failure[index]);  // an escaped poll(), once
            if (!job_error.empty()) {
                ctx.node_errors[index] = "node \"" + node.id + "\": " + job_error;
            } else if (!stage_error.empty()) {
                ctx.node_errors[index] = "node \"" + node.id + "\": " + stage_error;
            }
        }

        if (run.is_roi) {
            // Parity rule 1. ByOrigin is the shared comparator from
            // stage_graph.hpp - a second copy here could drift.
            std::stable_sort(results.begin(), results.end(), ByOrigin);
            ctx.report.roi_results[node.id] = results;
        } else if (job_error.empty() && !results.empty()) {
            ctx.report.node_results[node.id] = results[0];
            produced = results[0].data;
            ports = results[0].ports;
        }

        // Tracking runs once per producer per frame, before any consumer of
        // that producer is dispatched. ApplyTrackIds is the sole writer of
        // track_id; this mirrors SyncExecutor exactly.
        const BoxesData* boxes = NULL;
        std::shared_ptr<BoxesData> tracked;
        if (produced && node.tracked && ProducesRoi(produced->shape())) {
            const BoxesData* source_boxes =
                dynamic_cast<const BoxesData*>(produced.get());
            if (source_boxes != NULL) {
                tracked.reset(new BoxesData(*source_boxes));
                graph->FreshenStreamState(index, ctx.stream, ctx.generation);
                ApplyTrackIds(tracked.get(), node.trackers[ctx.stream].get());
                boxes = tracked.get();
                // Keep the origin the stage reported (its inv_align maps a
                // hand-off view back to the source frame) and its ports.
                StageResult result = ctx.report.node_results[node.id];  // origin and ports
                result.data = tracked;
                ctx.report.node_results[node.id] = result;
                produced = tracked;
            }
        } else if (produced && ProducesRoi(produced->shape())) {
            boxes = dynamic_cast<const BoxesData*>(produced.get());
        }

        ReleaseSuccessors(*graph, index, produced, boxes, ports, run.is_roi ? &results : NULL,
                          ctx);
    }

    /// A logic node's turn in this frame (its gate let it through): run it on
    /// the driver thread with this stream's state, record what it produced
    /// and release its successors.
    void HarvestLogic(FrameCtx& ctx, std::size_t index) {
        NodeRuntime& node = graph->mutable_nodes()[index];
        ctx.runs.erase(index);
        LogicFrame at;
        at.frame_size = ctx.frame.size();
        at.time_s = ctx.report.time_s;
        graph->FreshenStreamState(index, ctx.stream, ctx.generation);
        const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        const LogicOutput out = RunLogic(*node.logic, ctx.logic_in[index], at,
                                         node.logic_states[ctx.stream].get());
        ctx.report.node_ms[node.id] = ElapsedMs(started, std::chrono::steady_clock::now());
        ctx.report.node_results[node.id] = out.result;
        if (!IsEmpty(out.overlay)) ctx.report.overlays[node.id] = out.overlay;
        ctx.report.logic_nodes.insert(node.id);
        if (!out.events.empty()) ctx.node_events[index] = out.events;
        if (!out.error.empty() && !HasEarlierError(ctx, index)) {
            ctx.node_errors[index] = "node \"" + node.id + "\": " + out.error;
        }
        FrameView source;
        source.image = ctx.frame;
        ctx.view[index] = source;  // its boxes are in source pixels
        const BoxesData* boxes =
            out.result.data && ProducesRoi(out.result.data->shape())
                ? dynamic_cast<const BoxesData*>(out.result.data.get())
                : NULL;
        ReleaseSuccessors(*graph, index, out.result.data, boxes, StagePorts(), NULL, ctx);
    }

    /// True when no stream runs node `index`: reachable from no source,
    /// which only an unvalidated graph can have.
    bool InNoStream(std::size_t index) const {
        const std::vector<StreamPlan>& streams = graph->streams();
        for (std::size_t s = 0; s < streams.size(); ++s) {
            if (streams[s].member[index]) return false;
        }
        return true;
    }

    void FinishIfQuiet(const FrameCtxPtr& ctx) {
        if (ctx->closed || !ctx->ready.empty() || !ctx->runs.empty()) return;
        ctx->closed = true;
        // report.error holds one message. SyncExecutor overwrites it in
        // topological order, so replaying node errors in that order leaves
        // the same survivor whatever order the nodes ran in here. The
        // stream's order keeps the members' relative topological order.
        const StreamPlan& plan = graph->streams()[ctx->stream];
        const std::vector<std::size_t>& order = plan.order;
        for (std::size_t o = 0; o < order.size(); ++o) {
            std::map<std::size_t, std::string>::const_iterator it =
                ctx->node_errors.find(order[o]);
            if (it != ctx->node_errors.end()) ctx->report.error = it->second;
        }
        // A member never dispatched, or a node of no stream (no path from any
        // source), is something ValidateGraph rejects, so reaching this is a
        // scheduler defect - reported, never silent, and never on top of a
        // real inference error. A node of another stream only is not this
        // frame's to run, and is not reported.
        const std::vector<NodeRuntime>& nodes = graph->nodes();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if ((plan.member[i] || InNoStream(i)) && !ctx->done[i] &&
                ctx->report.error.empty()) {
                ctx->report.error = "node \"" + nodes[i].id + "\" never completed";
            }
        }
        // Unreachable through a validated graph - every member is dispatched -
        // but a tracked member this frame never reached would otherwise hold
        // every later frame of its stream at its gate forever. SkipGate may
        // harvest OTHER frames of the stream; it never touches this one,
        // which is closed.
        std::map<std::size_t, TrackGate>& stream_gates = gates[ctx->stream];
        for (std::map<std::size_t, TrackGate>::iterator g = stream_gates.begin();
             g != stream_gates.end(); ++g) {
            if (!ctx->done[g->first]) SkipGate(ctx->stream, g->first, ctx->stream_seq);
        }
        // Events in the stream's topological order, as SyncExecutor appends
        // them, whatever order the logic nodes ran in here.
        for (std::size_t o = 0; o < order.size(); ++o) {
            std::map<std::size_t, std::vector<GraphEvent> >::const_iterator raised =
                ctx->node_events.find(order[o]);
            if (raised == ctx->node_events.end()) continue;
            ctx->report.events.insert(ctx->report.events.end(), raised->second.begin(),
                                      raised->second.end());
        }
        Finished done;
        done.report = ctx->report;
        done.frame = ctx->frame;
        finished[ctx->seq] = done;
        frames.erase(ctx->seq);
    }

    /// Ask stages to deliver, one at a time, oldest outstanding submission
    /// first; process completions after each. See the header for why.
    bool PollOne() {
        std::vector<std::pair<std::size_t, std::size_t> > candidates;
        for (std::size_t i = 0; i < submit_ticks.size(); ++i) {
            if (!submit_ticks[i].empty()) {
                candidates.push_back(std::make_pair(submit_ticks[i].front(), i));
            }
        }
        std::sort(candidates.begin(), candidates.end());
        std::vector<NodeRuntime>& nodes = graph->mutable_nodes();
        for (std::size_t c = 0; c < candidates.size(); ++c) {
            const std::size_t index = candidates[c].second;
            try {
                nodes[index].stage->poll();
            } catch (const std::exception& error) {
                stage_failure[index] = error.what();
            } catch (...) {
                stage_failure[index] = "unknown error while polling stage";
            }
            if (DrainQueue()) return true;
        }
        return false;
    }

    bool Step(bool may_wait) {
        if (DrainQueue()) return true;
        if (PollOne()) return true;
        if (!may_wait) return false;
        queue->WaitFor(options.wait_slice_ms);
        return DrainQueue();
    }

    std::vector<std::string> StuckStages() const {
        std::vector<std::string> ids;
        if (graph == NULL) return ids;
        const std::vector<NodeRuntime>& nodes = graph->nodes();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (outstanding[i] > 0 || !backlog[i].empty()) ids.push_back(nodes[i].id);
        }
        return ids;
    }

    /// Admission control (Submit): true while another frame fits. The span
    /// runs from the oldest UNFINISHED frame to the next sequence number, so
    /// finished reports parked behind that frame count against the cap and
    /// a slow or lost head frame cannot let the source run on unbounded.
    /// Reports with nothing unfinished ahead of them do not count.
    bool HasRoomForAnotherFrame() const {
        return frames.empty() ||
               next_seq - frames.begin()->first < options.max_frames_in_flight;
    }

    template <typename Done>
    void PumpUntil(Done done) {
        typedef std::chrono::steady_clock Clock;
        Clock::time_point last_progress = Clock::now();
        while (!done()) {
            if (Step(true)) {
                last_progress = Clock::now();
                continue;
            }
            const std::vector<std::string> stuck = StuckStages();
            if (stuck.empty()) {
                // Frames in flight with nothing outstanding anywhere can only
                // be a scheduler defect; spinning on it would hide it.
                throw std::logic_error(
                    "AsyncExecutor: frames are in flight but no stage has "
                    "outstanding work");
            }
            // Name the stuck stages only once the executor has been quiet for
            // stuck_report_ms. At the first quiet slice every healthy stage
            // is mid-job - a slice is 5 ms, a hardware job about 30 - so
            // reporting then names the healthy stages and misses the lost one.
            if (!stuck_reported && options.interrupted && options.interrupted() &&
                Clock::now() - last_progress >=
                    std::chrono::milliseconds(options.stuck_report_ms)) {
                stuck_reported = true;
                if (options.on_stuck) options.on_stuck(stuck);
            }
            if (options.stall_timeout_ms != 0 &&
                Clock::now() - last_progress > StallTimeout(options.stall_timeout_ms)) {
                throw std::runtime_error(
                    "AsyncExecutor stalled: no completion for " +
                    std::to_string(options.stall_timeout_ms) +
                    " ms; waiting on " + JoinIds(stuck));
            }
        }
    }

    /// Flush every stage of `target` - including work no frame submitted -
    /// and process anything that produced. A throwing flush() has nowhere
    /// to be reported here and must not escape a caller that is draining
    /// precisely because something already went wrong.
    void FlushAll(StageGraph& target) {
        std::vector<NodeRuntime>& nodes = target.mutable_nodes();
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].stage) continue;
            try {
                nodes[i].stage->flush();
            } catch (...) {
            }
        }
        if (graph == &target) DrainQueue();
    }
};

AsyncExecutor::AsyncExecutor(const AsyncOptions& options)
    : engine_(new Engine(options)) {
    if (options.max_frames_in_flight == 0) {
        throw std::invalid_argument(
            "AsyncOptions::max_frames_in_flight must be at least 1");
    }
    if (options.wait_slice_ms == 0) {
        throw std::invalid_argument("AsyncOptions::wait_slice_ms must be at least 1");
    }
}

AsyncExecutor::~AsyncExecutor() {}

const AsyncOptions& AsyncExecutor::options() const { return engine_->options; }

void AsyncExecutor::Submit(StageGraph& graph, const cv::Mat& frame, std::size_t frame_index) {
    Submit(graph, graph.OnlyStream("AsyncExecutor::Submit"), frame, frame_index);
}

void AsyncExecutor::Submit(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                           std::size_t frame_index) {
    if (stream >= graph.streams().size()) {
        throw std::out_of_range("AsyncExecutor::Submit: stream " + std::to_string(stream) +
                                " of a graph with " +
                                std::to_string(graph.streams().size()) + " stream(s)");
    }
    Engine* engine = engine_.get();
    engine->Bind(graph);
    engine->PumpUntil([engine] { return engine->HasRoomForAnotherFrame(); });
    engine->Admit(frame, stream, frame_index);
}

bool AsyncExecutor::TryNext(FrameReport* report, cv::Mat* frame) {
    Engine* engine = engine_.get();
    // Nothing can be outstanding once every admitted frame has finished, so
    // there is nothing to process - and the bound graph may already be gone.
    if (engine->graph != NULL && !engine->frames.empty()) {
        while (engine->Step(false)) {
        }
    }
    std::map<std::size_t, Finished>::iterator it =
        engine->finished.find(engine->next_emit);
    if (it == engine->finished.end()) return false;
    *report = it->second.report;
    if (frame != NULL) *frame = it->second.frame;
    engine->finished.erase(it);
    ++engine->next_emit;
    return true;
}

void AsyncExecutor::Finish(StageGraph& graph) {
    Engine* engine = engine_.get();
    if (!engine->frames.empty()) {
        engine->PumpUntil([engine] { return engine->frames.empty(); });
    }
    engine->FlushAll(graph);
}

std::size_t AsyncExecutor::frames_in_flight() const {
    return engine_->frames.size();
}

std::size_t AsyncExecutor::peak_frames_in_flight() const {
    return engine_->peak_in_flight;
}

FrameReport AsyncExecutor::RunFrame(StageGraph& graph, const cv::Mat& frame,
                                    std::size_t frame_index) {
    return RunFrame(graph, graph.OnlyStream("AsyncExecutor::RunFrame"), frame, frame_index);
}

FrameReport AsyncExecutor::RunFrame(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                                    std::size_t frame_index) {
    Engine* engine = engine_.get();
    if (!engine->frames.empty() || !engine->finished.empty()) {
        throw std::logic_error(
            "AsyncExecutor::RunFrame: frames from Submit() are still in "
            "flight or their reports are untaken; call Finish() and drain "
            "with TryNext() first");
    }
    Submit(graph, stream, frame, frame_index);
    // Finish also flushes every stage, so nothing is left outstanding when
    // the next frame starts, including work on a stage this frame never
    // dispatched (TestAsyncDrainsNeverDispatchedStage).
    Finish(graph);
    FrameReport report;
    if (!TryNext(&report)) {
        // Finish returned with every admitted frame finished and nothing
        // was untaken before Submit, so the one report must be there.
        throw std::logic_error(
            "AsyncExecutor::RunFrame: the frame finished but left no report");
    }
    return report;
}

void AsyncExecutor::DrainPending(StageGraph& graph) { Finish(graph); }

}  // namespace graph
}  // namespace dxapp
