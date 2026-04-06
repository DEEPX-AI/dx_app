/**
 * @file deepmar_resnet50_sync.cpp
 * @brief Deepmar_ResNet50 synchronous classification example using SyncClassificationRunner
 */

#include "factory/deepmar_resnet50_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deepmar_ResNet50Factory>();
    dxapp::SyncClassificationRunner<dxapp::Deepmar_ResNet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
