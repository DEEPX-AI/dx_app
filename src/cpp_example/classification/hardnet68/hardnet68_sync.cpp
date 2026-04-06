/**
 * @file hardnet68_sync.cpp
 * @brief Hardnet68 synchronous classification example using SyncClassificationRunner
 */

#include "factory/hardnet68_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Hardnet68Factory>();
    dxapp::SyncClassificationRunner<dxapp::Hardnet68Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
