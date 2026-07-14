/**
 * @file deitbase384_sync.cpp
 * @brief Deitbase384 synchronous classification example using SyncClassificationRunner
 */

#include "factory/deitbase384_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deitbase384Factory>();
    dxapp::SyncClassificationRunner<dxapp::Deitbase384Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
