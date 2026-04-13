/**
 * @file levit192_hug_sync.cpp
 * @brief Levit192_hug synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit192_hug_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit192_hugFactory>();
    dxapp::SyncClassificationRunner<dxapp::Levit192_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
