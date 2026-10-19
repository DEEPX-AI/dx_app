/**
 * @file multi_model_graph_async.cpp
 * @brief Run a node-graph JSON with the dispatch-on-ready executor.
 *
 * Identical to multi_model_graph_sync.cpp except for the executor kind.
 * Every sibling node whose inputs are ready is submitted before any of
 * them is waited on, and several frames are processed at once:
 * --max-inflight counts the frames in flight (default 16, at least 1).
 * Reports still come out in frame order and match the sync executor's.
 */
#include "multi_model_graph/graph_cli.hpp"

int main(int argc, char** argv) {
    return dxapp::graph::cli::Main(argc, argv,
                                   dxapp::graph::cli::kAsyncExecutor);
}
