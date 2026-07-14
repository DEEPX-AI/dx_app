/**
 * @file levit384_sync.cpp
 * @brief Levit384 synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit384_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit384Factory>();
    dxapp::SyncClassificationRunner<dxapp::Levit384Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
