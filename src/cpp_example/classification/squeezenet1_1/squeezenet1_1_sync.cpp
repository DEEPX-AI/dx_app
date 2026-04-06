/**
 * @file squeezenet1_1_sync.cpp
 * @brief Squeezenet1_1 synchronous classification example using SyncClassificationRunner
 */

#include "factory/squeezenet1_1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Squeezenet1_1Factory>();
    dxapp::SyncClassificationRunner<dxapp::Squeezenet1_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
