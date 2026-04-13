/**
 * @file efficientdetd4_sync.cpp
 * @brief Efficientdetd4 synchronous inference example
 */

#include "factory/efficientdetd4_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientdetd4Factory>();
    dxapp::SyncDetectionRunner<dxapp::Efficientdetd4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
