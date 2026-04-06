/**
 * @file wideresnet50_2_sync.cpp
 * @brief WideResNet50_2 synchronous classification example using SyncClassificationRunner
 */

#include "factory/wideresnet50_2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::WideResNet50_2Factory>();
    dxapp::SyncClassificationRunner<dxapp::WideResNet50_2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
