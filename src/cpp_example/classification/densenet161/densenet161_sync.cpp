/**
 * @file densenet161_sync.cpp
 * @brief Densenet161 synchronous classification example using SyncClassificationRunner
 */

#include "factory/densenet161_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet161Factory>();
    dxapp::SyncClassificationRunner<dxapp::Densenet161Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
