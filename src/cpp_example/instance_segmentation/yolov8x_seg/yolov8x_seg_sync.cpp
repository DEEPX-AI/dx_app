/**
 * @file yolov8x_seg_sync.cpp
 * @brief Yolov8xSegFactory synchronous inference example
 */

#include "factory/yolov8x_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8xSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolov8xSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
