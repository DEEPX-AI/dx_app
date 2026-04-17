/**
 * @file regnetx400mf_sync.cpp
 * @brief Regnetx400mf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx400mf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx400mfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx400mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
