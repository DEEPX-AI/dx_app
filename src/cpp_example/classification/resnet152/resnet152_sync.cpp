/**
 * @file resnet152_sync.cpp
 * @brief ResNet152 synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnet152_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet152Factory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet152Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
