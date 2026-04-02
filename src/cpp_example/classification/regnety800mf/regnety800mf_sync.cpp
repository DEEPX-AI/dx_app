/**
 * @file regnety800mf_sync.cpp
 * @brief Regnety800mf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety800mf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety800mfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety800mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
