/**
 * @file levit128_sync.cpp
 * @brief Levit128 synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit128_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit128Factory>();
    dxapp::SyncClassificationRunner<dxapp::Levit128Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
