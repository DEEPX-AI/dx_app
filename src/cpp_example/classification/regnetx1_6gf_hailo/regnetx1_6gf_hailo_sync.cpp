/**
 * @file regnetx1_6gf_hailo_sync.cpp
 * @brief Regnetx1_6gf_hailo synchronous classification example using SyncClassificationRunner
 */

#include "factory/regnetx1_6gf_hailo_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx1_6gf_hailoFactory>();
    dxapp::SyncClassificationRunner<dxapp::Regnetx1_6gf_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
