/**
 * @file yolov8l_seg_sync.cpp
 * @brief Yolov8lSegFactory synchronous inference example
 */

#include "factory/yolov8l_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8lSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolov8lSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
