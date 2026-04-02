/**
 * @file efficientnet_lite3_sync.cpp
 * @brief EfficientNet_lite3 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnet_lite3_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite3Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNet_lite3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
