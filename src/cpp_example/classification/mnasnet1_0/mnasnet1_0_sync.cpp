/**
 * @file mnasnet1_0_sync.cpp
 * @brief Mnasnet1_0 synchronous classification example using SyncClassificationRunner
 */

#include "factory/mnasnet1_0_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet1_0Factory>();
    dxapp::SyncClassificationRunner<dxapp::Mnasnet1_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
