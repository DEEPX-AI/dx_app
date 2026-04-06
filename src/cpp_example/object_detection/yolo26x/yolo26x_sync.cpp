/**
 * @file yolo26x_sync.cpp
 * @brief Yolo26x synchronous inference example
 */

#include "factory/yolo26x_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26xFactory>();
    dxapp::SyncDetectionRunner<dxapp::Yolo26xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
