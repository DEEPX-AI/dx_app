/**
 * @file resnet18_brecq_sync.cpp
 * @brief ResNet18_brecq synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnet18_brecq_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ResNet18_brecqFactory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet18_brecqFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
