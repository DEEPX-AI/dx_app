/**
 * @file yolo26s_sync.cpp
 * @brief Yolo26s synchronous inference example
 */

#include "factory/yolo26s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26sFactory>();
    dxapp::SyncDetectionRunner<dxapp::Yolo26sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
