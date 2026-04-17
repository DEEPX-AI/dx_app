/**
 * @file levit128_hug_sync.cpp
 * @brief Levit128_hug synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit128_hug_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit128_hugFactory>();
    dxapp::SyncClassificationRunner<dxapp::Levit128_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
