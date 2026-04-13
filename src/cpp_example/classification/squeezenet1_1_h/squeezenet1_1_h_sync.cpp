/**
 * @file squeezenet1_1_h_sync.cpp
 * @brief Squeezenet1_1_h synchronous classification example using SyncClassificationRunner
 */

#include "factory/squeezenet1_1_h_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Squeezenet1_1_hFactory>();
    dxapp::SyncClassificationRunner<dxapp::Squeezenet1_1_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
