/**
 * @file resnet34_sync.cpp
 * @brief ResNet34 synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnet34_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet34Factory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet34Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
