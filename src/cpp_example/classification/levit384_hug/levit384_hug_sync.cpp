/**
 * @file levit384_hug_sync.cpp
 * @brief Levit384_hug synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit384_hug_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit384_hugFactory>();
    dxapp::SyncClassificationRunner<dxapp::Levit384_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
