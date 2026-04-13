/**
 * @file efficientdetd2_sync.cpp
 * @brief Efficientdetd2 synchronous inference example
 */

#include "factory/efficientdetd2_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientdetd2Factory>();
    dxapp::SyncDetectionRunner<dxapp::Efficientdetd2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
