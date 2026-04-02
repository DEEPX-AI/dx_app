/**
 * @file yolo26n_sync.cpp
 * @brief Yolo26n synchronous inference example
 */

#include "factory/yolo26n_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26nFactory>();
    dxapp::SyncDetectionRunner<dxapp::Yolo26nFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
