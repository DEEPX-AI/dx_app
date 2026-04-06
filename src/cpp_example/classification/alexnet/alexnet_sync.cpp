/**
 * @file alexnet_sync.cpp
 * @brief Alexnet synchronous classification example using SyncClassificationRunner
 */

#include "factory/alexnet_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::AlexnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::AlexnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
