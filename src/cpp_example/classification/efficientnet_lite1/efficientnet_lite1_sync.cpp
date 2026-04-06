/**
 * @file efficientnet_lite1_sync.cpp
 * @brief EfficientNet_lite1 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnet_lite1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite1Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNet_lite1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
