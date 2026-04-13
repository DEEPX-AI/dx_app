/**
 * @file yoloxx_sync.cpp
 * @brief YOLOXx synchronous inference example
 */

#include "factory/yoloxx_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXxFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOXxFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
