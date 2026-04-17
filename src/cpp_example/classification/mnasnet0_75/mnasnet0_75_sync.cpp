/**
 * @file mnasnet0_75_sync.cpp
 * @brief Mnasnet0_75 synchronous classification example using SyncClassificationRunner
 */

#include "factory/mnasnet0_75_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet0_75Factory>();
    dxapp::SyncClassificationRunner<dxapp::Mnasnet0_75Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
