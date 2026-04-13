/**
 * @file efficientdetd1_sync.cpp
 * @brief Efficientdetd1 synchronous inference example
 */

#include "factory/efficientdetd1_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientdetd1Factory>();
    dxapp::SyncDetectionRunner<dxapp::Efficientdetd1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
