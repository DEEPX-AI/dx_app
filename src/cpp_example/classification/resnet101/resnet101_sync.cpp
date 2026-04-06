/**
 * @file resnet101_sync.cpp
 * @brief ResNet101 synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnet101_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet101Factory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet101Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
