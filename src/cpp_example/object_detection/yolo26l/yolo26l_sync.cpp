/**
 * @file yolo26l_sync.cpp
 * @brief Yolo26l synchronous inference example
 */

#include "factory/yolo26l_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26lFactory>();
    dxapp::SyncDetectionRunner<dxapp::Yolo26lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
