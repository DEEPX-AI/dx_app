/**
 * @file efficientnetb2_sync.cpp
 * @brief EfficientNetb2 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetb2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb2Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetb2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
