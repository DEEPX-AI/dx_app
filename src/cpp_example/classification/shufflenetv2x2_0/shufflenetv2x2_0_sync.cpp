/**
 * @file shufflenetv2x2_0_sync.cpp
 * @brief Shufflenetv2x2_0 synchronous classification example using SyncClassificationRunner
 */

#include "factory/shufflenetv2x2_0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Shufflenetv2x2_0Factory>();
    dxapp::SyncClassificationRunner<dxapp::Shufflenetv2x2_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
