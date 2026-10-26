/**
 * @file multi_model_graph_sync.cpp
 * @brief Run a node-graph JSON with the synchronous executor.
 *
 * Identical to multi_model_graph_async.cpp except for the executor kind.
 * Slower by design: it runs one stage at a time in topological order,
 * which makes it the reference the asynchronous executor is compared
 * against and the first thing to try when a result looks wrong.
 */
#include "multi_model_graph/graph_cli.hpp"

int main(int argc, char** argv) {
    return dxapp::graph::cli::Main(argc, argv,
                                   dxapp::graph::cli::kSyncExecutor);
}
