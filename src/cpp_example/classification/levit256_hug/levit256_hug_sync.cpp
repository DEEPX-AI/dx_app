/**
 * @file levit256_hug_sync.cpp
 * @brief Levit256_hug synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit256_hug_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit256_hugFactory>();
    dxapp::SyncClassificationRunner<dxapp::Levit256_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
