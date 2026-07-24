/**
 * @file levit256_sync.cpp
 * @brief Levit256 synchronous classification example using SyncClassificationRunner
 */

#include "factory/levit256_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit256Factory>();
    dxapp::SyncClassificationRunner<dxapp::Levit256Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
