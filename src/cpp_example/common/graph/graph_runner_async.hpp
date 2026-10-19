/**
 * @file graph_runner_async.hpp
 * @brief Dispatch-on-ready executor, harvesting on completion.
 *
 * Every job's callback records its result into that job's collector and
 * pushes a notice into a completion queue this executor owns. One driver
 * thread - the caller's - drains the queue: a node whose last job has come
 * back is harvested, its successors are released, and whatever became ready
 * is submitted at once. Nothing waits on flush(). flush() waits for EVERY
 * job outstanding on a stage, so a flush-driven harvest cannot tell one
 * job's completion from another's; completion-driven harvesting can, which
 * is what lets this executor keep several frames on one stage at once.
 *
 * When the queue is empty the driver asks one stage at a time to deliver
 * what it holds (IStage::poll(), clause (6)) - oldest outstanding
 * submission first - and processes completions after each. On real
 * hardware poll() is a no-op and callbacks arrive on the runtime's own
 * thread; with the test fake, poll() is what delivers, and polling one
 * stage at a time is what keeps a cascade child submitted while its
 * parent's siblings are still outstanding (TestAsyncOverlapsAcrossDepth).
 * Polling every stage at once would deliver the siblings together and the
 * overlap would vanish from the measurement, not from the hardware.
 *
 * If nothing is deliverable the driver waits on the queue in slices of
 * AsyncOptions::wait_slice_ms, checking the interrupt hook between slices.
 *
 * The three parity rules (spec 4.5) that make this agree with SyncExecutor:
 *
 *  1. Completions arrive in whatever order the runtime hands them back, so a
 *     node's ROI results are stable_sorted into ByOrigin order - the single
 *     comparator declared in stage_graph.hpp and shared with SyncExecutor.
 *  2. Trackers and logic nodes are stateful and must see frames in order.
 *     Each (stream, tracked or logic node) pair has a gate holding the next
 *     frame of that stream it may run, counted in that stream's admission
 *     order; a completion for a later frame of the stream (for a logic node:
 *     its dispatch) is parked until every earlier frame of the stream has
 *     passed - harvested, or skipped because it never reached that node (an
 *     empty frame). Frames of other streams never touch it. Only this CPU
 *     step is serialized; later frames' inference keeps running.
 *     ApplyTrackIds stays the sole writer of track_id in the tree. A logic
 *     node runs on the driver thread at its turn (HarvestLogic); its events
 *     join the report in topological order when the frame finishes.
 *  3. Nothing here accumulates floating point across stages.
 *
 * Reports are emitted in admission order: a frame that finishes early waits
 * in a buffer until every earlier frame has been handed out. That buffer is
 * inside the frame cap: a finished report still waiting behind an unfinished
 * frame counts as in flight, so a slow or lost head frame stops admission
 * instead of letting the source be read without bound (see Submit()).
 * The window is shared by every stream: the span runs over admission order
 * whatever the stream, and reports come out in admission order (SP2 R1, R2).
 *
 * report.error holds one message; errors are recorded per node and resolved
 * in topological order when the frame finishes, as SyncExecutor does.
 */
#ifndef DXAPP_GRAPH_GRAPH_RUNNER_ASYNC_HPP
#define DXAPP_GRAPH_GRAPH_RUNNER_ASYNC_HPP

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/graph/stage_graph.hpp"

namespace dxapp {
namespace graph {

struct AsyncOptions {
    /// Frames in flight (spec 4.5: "--max-inflight frames"): the span of
    /// admission sequence numbers from the oldest unfinished frame to the
    /// newest admitted one, so a finished report held behind an unfinished
    /// frame counts too. Must be at least 1. Default 16, from the spec P5
    /// sweep.
    std::size_t max_frames_in_flight;
    /// Jobs one stage may have outstanding, counted across frames. 0 means
    /// no limit.
    std::size_t max_jobs_per_stage;
    /// Wait slice between poll() / interrupt checks when nothing is
    /// deliverable. Must be at least 1.
    std::size_t wait_slice_ms;
    /// Throw std::runtime_error when work is outstanding and nothing has
    /// completed for this long. 0 waits forever - the production default,
    /// because dxrt cannot cancel an in-flight job (spec P7). Tests set it
    /// so a regression fails instead of hanging.
    std::size_t stall_timeout_ms;
    /// How long the executor must have processed no completion, while
    /// interrupted() is true, before on_stuck names the stages still holding
    /// work. Long enough that a healthy stage - tens of ms per job on real
    /// hardware - delivers first, so what is named is what is actually lost
    /// rather than whatever happened to be mid-job. 0 reports at the first
    /// quiet wait slice. Default 1000.
    std::size_t stuck_report_ms;
    /// Optional. Checked between wait slices.
    std::function<bool()> interrupted;
    /// Optional. Called at most ONCE per executor, with the id of every node
    /// whose stage still has outstanding or queued jobs, when interrupted()
    /// is true AND no completion has been processed for stuck_report_ms.
    std::function<void(const std::vector<std::string>&)> on_stuck;

    AsyncOptions()
        : max_frames_in_flight(16), max_jobs_per_stage(0), wait_slice_ms(5),
          stall_timeout_ms(0), stuck_report_ms(1000) {}
};

/**
 * AFTER AN EXCEPTION, DESTROY THE EXECUTOR. Once Submit(), Finish(),
 * RunFrame() or TryNext() has thrown - a stall timeout, a scheduler
 * defect, a misuse logic_error - the executor may hold frames whose jobs
 * will never be delivered, so no further call on it is meaningful. Destroy
 * it (and build a new one if the caller carries on). Destruction drops any
 * untaken reports and does not wait for outstanding jobs: the stages wait
 * for their own jobs when THEY are destroyed (IStage clause (5)), and every
 * completion callback writes only into state it co-owns, so a completion
 * that arrives after the executor is gone is harmless.
 */
class AsyncExecutor {
 public:
    /// @throws std::invalid_argument if max_frames_in_flight or
    ///         wait_slice_ms is 0.
    explicit AsyncExecutor(const AsyncOptions& options);
    ~AsyncExecutor();

    /// One frame, start to finish. Throws std::logic_error if called while
    /// frames from Submit() are in flight or their reports are untaken
    /// (call Finish() and drain with TryNext() first), and if the frame
    /// finishes without leaving a report - a scheduler defect, never an
    /// empty report passed off as a result.
    FrameReport RunFrame(StageGraph& graph, const cv::Mat& frame,
                         std::size_t frame_index);

    /// Admit a frame (copied - sources reuse their buffer). A frame is
    /// admitted only while fewer than max_frames_in_flight admissions
    /// separate it from the oldest UNFINISHED frame; a finished report
    /// waiting behind that frame counts, so a slow or lost head frame holds
    /// at most max_frames_in_flight frames (and their copies) rather than
    /// the whole source. When full, processes completions until the oldest
    /// unfinished frame finishes - throwing on stall_timeout_ms - and
    /// otherwise returns without waiting.
    ///
    /// Reports with no unfinished frame ahead of them do not count, so a
    /// caller that submits every frame and drains only afterwards still
    /// runs; such a caller holds every report, with its frame copy, until it
    /// drains. A streaming caller calls TryNext() after each Submit().
    void Submit(StageGraph& graph, const cv::Mat& frame, std::size_t frame_index);

    /// One frame of stream `stream` (an index into graph.streams()) - the
    /// stream's nodes only, with its own trackers and tracker gates. The frame
    /// window (max_frames_in_flight) and the report order are shared by every
    /// stream: reports come out in admission order, whatever the stream.
    /// Throws std::out_of_range for a stream the graph does not have.
    void Submit(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                std::size_t frame_index);
    FrameReport RunFrame(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                         std::size_t frame_index);
    // The three-argument Submit/RunFrame stay: the graph's only stream
    // (StageGraph::OnlyStream throws std::invalid_argument for several).

    /// Process whatever has completed, without waiting, then hand out the
    /// next report IN FRAME ORDER if it is finished. `frame`, if given,
    /// receives the frame that report belongs to. Reports accumulate until
    /// the caller takes them: one finished behind an unfinished frame
    /// counts against max_frames_in_flight, one with nothing unfinished
    /// ahead of it is simply held (see Submit()).
    bool TryNext(FrameReport* report, cv::Mat* frame = NULL);

    /// Process until every admitted frame has finished, then flush every
    /// stage. Reports stay available to TryNext.
    void Finish(StageGraph& graph);

    std::size_t frames_in_flight() const;
    std::size_t peak_frames_in_flight() const;

    /// Finish every frame in flight, then flush every stage of the graph so
    /// no work - including work no frame submitted - is left outstanding.
    void DrainPending(StageGraph& graph);

    const AsyncOptions& options() const;

 private:
    AsyncExecutor(const AsyncExecutor&) = delete;
    AsyncExecutor& operator=(const AsyncExecutor&) = delete;

    struct Engine;
    std::unique_ptr<Engine> engine_;
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_GRAPH_RUNNER_ASYNC_HPP
