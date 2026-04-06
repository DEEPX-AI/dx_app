/**
 * @file regnetx800mf_sync.cpp
 * @brief Regnetx800mf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx800mf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx800mfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx800mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
