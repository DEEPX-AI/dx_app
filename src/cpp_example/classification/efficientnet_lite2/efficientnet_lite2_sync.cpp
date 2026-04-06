/**
 * @file efficientnet_lite2_sync.cpp
 * @brief EfficientNet_lite2 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnet_lite2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite2Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNet_lite2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
