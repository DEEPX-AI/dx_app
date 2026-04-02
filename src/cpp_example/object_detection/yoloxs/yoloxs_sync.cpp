/**
 * @file yoloxs_sync.cpp
 * @brief YOLOX synchronous inference example
 */

#include "factory/yoloxs_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOXFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
