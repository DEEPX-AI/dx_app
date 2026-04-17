/**
 * @file yoloxl_sync.cpp
 * @brief YOLOXl synchronous inference example
 */

#include "factory/yoloxl_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXlFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOXlFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
