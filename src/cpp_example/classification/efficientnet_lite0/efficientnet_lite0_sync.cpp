/**
 * @file efficientnet_lite0_sync.cpp
 * @brief EfficientNet synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnet_lite0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetFactory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
