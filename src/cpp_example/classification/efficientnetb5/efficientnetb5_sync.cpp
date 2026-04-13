/**
 * @file efficientnetb5_sync.cpp
 * @brief EfficientNetb5 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetb5_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb5Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetb5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
