/**
 * @file yoloxtiny_sync.cpp
 * @brief YOLOXtiny synchronous inference example
 */

#include "factory/yoloxtiny_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXtinyFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOXtinyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
