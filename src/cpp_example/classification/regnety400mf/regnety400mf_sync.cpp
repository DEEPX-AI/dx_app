/**
 * @file regnety400mf_sync.cpp
 * @brief Regnety400mf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety400mf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety400mfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety400mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
