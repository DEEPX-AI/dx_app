/**
 * @file mnasnet0_5_sync.cpp
 * @brief Mnasnet0_5 synchronous classification example using SyncClassificationRunner
 */

#include "factory/mnasnet0_5_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet0_5Factory>();
    dxapp::SyncClassificationRunner<dxapp::Mnasnet0_5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
