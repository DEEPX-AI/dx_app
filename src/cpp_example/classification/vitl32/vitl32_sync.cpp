/**
 * @file vitl32_sync.cpp
 * @brief Vitl32 synchronous classification example using SyncClassificationRunner
 */

#include "factory/vitl32_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vitl32Factory>();
    dxapp::SyncClassificationRunner<dxapp::Vitl32Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
