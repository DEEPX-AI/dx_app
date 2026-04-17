/**
 * @file densenet121_sync.cpp
 * @brief Densenet121 synchronous classification example using SyncClassificationRunner
 */

#include "factory/densenet121_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet121Factory>();
    dxapp::SyncClassificationRunner<dxapp::Densenet121Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
