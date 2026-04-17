/**
 * @file hardnet39ds_sync.cpp
 * @brief Hardnet39ds synchronous classification example using SyncClassificationRunner
 */

#include "factory/hardnet39ds_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Hardnet39dsFactory>();
    dxapp::SyncClassificationRunner<dxapp::Hardnet39dsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
