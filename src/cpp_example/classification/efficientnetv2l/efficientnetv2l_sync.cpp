/**
 * @file efficientnetv2l_sync.cpp
 * @brief EfficientNetv2l synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetv2l_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetv2lFactory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetv2lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
