/**
 * @file regnety200mf_sync.cpp
 * @brief Regnety200mf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnety200mf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety200mfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnety200mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
