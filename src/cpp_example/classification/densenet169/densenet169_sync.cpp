/**
 * @file densenet169_sync.cpp
 * @brief Densenet169 synchronous classification example using SyncClassificationRunner
 */

#include "factory/densenet169_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet169Factory>();
    dxapp::SyncClassificationRunner<dxapp::Densenet169Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
