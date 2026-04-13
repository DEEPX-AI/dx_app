/**
 * @file shufflenetv1x1_0_sync.cpp
 * @brief Shufflenetv1x1_0 synchronous classification example using SyncClassificationRunner
 */

#include "factory/shufflenetv1x1_0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Shufflenetv1x1_0Factory>();
    dxapp::SyncClassificationRunner<dxapp::Shufflenetv1x1_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
