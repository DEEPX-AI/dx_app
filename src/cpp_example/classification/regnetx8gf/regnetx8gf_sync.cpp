/**
 * @file regnetx8gf_sync.cpp
 * @brief Regnetx8gf synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx8gf_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx8gfFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx8gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
