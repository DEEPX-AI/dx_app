/**
 * @file deitbase384_hug_sync.cpp
 * @brief Deitbase384_hug synchronous classification example using SyncClassificationRunner
 */

#include "factory/deitbase384_hug_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deitbase384_hugFactory>();
    dxapp::SyncClassificationRunner<dxapp::Deitbase384_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
