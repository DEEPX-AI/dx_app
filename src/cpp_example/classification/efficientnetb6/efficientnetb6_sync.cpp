/**
 * @file efficientnetb6_sync.cpp
 * @brief EfficientNetb6 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetb6_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb6Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetb6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
