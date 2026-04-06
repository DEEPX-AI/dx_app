/**
 * @file resnet18_sync.cpp
 * @brief ResNet18 synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnet18_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet18Factory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
