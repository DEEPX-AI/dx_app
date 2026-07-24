/**
 * @file levit192_sync.cpp
 * @brief Levit192 synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit192_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit192Factory>();
    dxapp::SyncClassificationRunner<dxapp::Levit192Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
