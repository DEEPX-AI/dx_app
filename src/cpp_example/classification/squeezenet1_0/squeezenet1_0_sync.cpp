/**
 * @file squeezenet1_0_sync.cpp
 * @brief Squeezenet1_0 synchronous classification example using SyncClassificationRunner
 */

#include "factory/squeezenet1_0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Squeezenet1_0Factory>();
    dxapp::SyncClassificationRunner<dxapp::Squeezenet1_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
