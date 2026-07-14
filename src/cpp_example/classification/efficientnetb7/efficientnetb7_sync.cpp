/**
 * @file efficientnetb7_sync.cpp
 * @brief Efficientnetb7Factory synchronous inference example
 */

#include "factory/efficientnetb7_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientnetb7Factory>();
    dxapp::SyncClassificationRunner<dxapp::Efficientnetb7Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
