/**
 * @file regnetx32gf_sync.cpp
 * @brief Regnetx32gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx32gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx32gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx32gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
