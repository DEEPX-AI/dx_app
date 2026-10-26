#ifndef DXAPP_GRAPH_GRAPH_RUNNER_SYNC_HPP
#define DXAPP_GRAPH_GRAPH_RUNNER_SYNC_HPP

#include "common/graph/stage_graph.hpp"

namespace dxapp {
namespace graph {

/**
 * @brief Topological-order executor: one stage at a time.
 *
 * Slower than async by design. Its value is determinism: it is the reference
 * the async executor is compared against, and the first thing to try when a
 * result looks wrong.
 */
class SyncExecutor {
 public:
    /// One frame of the graph's only stream (OnlyStream throws for several).
    FrameReport RunFrame(StageGraph& graph, const cv::Mat& frame, std::size_t frame_index);
    /// One frame of stream `stream` (an index into graph.streams()): the
    /// stream's nodes, in topological order, with its own trackers. Throws
    /// std::out_of_range for a stream the graph does not have.
    FrameReport RunFrame(StageGraph& graph, std::size_t stream, const cv::Mat& frame,
                         std::size_t frame_index);
};

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_GRAPH_RUNNER_SYNC_HPP
