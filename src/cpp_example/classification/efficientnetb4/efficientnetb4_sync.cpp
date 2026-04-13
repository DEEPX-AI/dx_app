/**
 * @file efficientnetb4_sync.cpp
 * @brief EfficientNetb4 synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientnetb4_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EfficientNetb4Factory>();
    dxapp::SyncClassificationRunner<dxapp::EfficientNetb4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
