/**
 * @file wideresnet101_2_sync.cpp
 * @brief WideResNet101_2 synchronous classification example using SyncClassificationRunner
 */

#include "factory/wideresnet101_2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::WideResNet101_2Factory>();
    dxapp::SyncClassificationRunner<dxapp::WideResNet101_2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
