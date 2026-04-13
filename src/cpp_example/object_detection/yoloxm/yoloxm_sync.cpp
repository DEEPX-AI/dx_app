/**
 * @file yoloxm_sync.cpp
 * @brief YOLOXm synchronous inference example
 */

#include "factory/yoloxm_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXmFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOXmFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
