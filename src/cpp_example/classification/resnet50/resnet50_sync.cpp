/**
 * @file resnet50_sync.cpp
 * @brief ResNet50 synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnet50_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet50Factory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
