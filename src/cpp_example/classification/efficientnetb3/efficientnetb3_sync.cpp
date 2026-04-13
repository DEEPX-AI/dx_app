/**
 * @file efficientnetb3_sync.cpp
 * @brief EfficientNetb3 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetb3_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb3Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetb3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
