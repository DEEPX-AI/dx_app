/**
 * @file efficientnetv2s_sync.cpp
 * @brief EfficientNetv2s synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetv2s_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetv2sFactory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetv2sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
