/**
 * @file efficientformerv2_l_sync.cpp
 * @brief Efficientformerv2_l synchronous classification example using SyncClassificationRunner
 */

#include "factory/efficientformerv2_l_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientformerv2_lFactory>();
    dxapp::SyncClassificationRunner<dxapp::Efficientformerv2_lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
