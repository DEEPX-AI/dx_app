/**
 * @file mnasnet1_3_sync.cpp
 * @brief Mnasnet1_3 synchronous classification example using SyncClassificationRunner
 */

#include "factory/mnasnet1_3_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet1_3Factory>();
    dxapp::SyncClassificationRunner<dxapp::Mnasnet1_3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
