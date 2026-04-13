/**
 * @file densenet201_sync.cpp
 * @brief Densenet201 synchronous classification example using SyncClassificationRunner
 */

#include "factory/densenet201_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet201Factory>();
    dxapp::SyncClassificationRunner<dxapp::Densenet201Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
