/**
 * @file efficientnet_lite4_sync.cpp
 * @brief EfficientNet_lite4 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnet_lite4_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNet_lite4Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNet_lite4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
