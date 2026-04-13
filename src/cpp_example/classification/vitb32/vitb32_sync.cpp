/**
 * @file vitb32_sync.cpp
 * @brief Vitb32 synchronous classification example using SyncClassificationRunner
 */

#include "factory/vitb32_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vitb32Factory>();
    dxapp::SyncClassificationRunner<dxapp::Vitb32Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
