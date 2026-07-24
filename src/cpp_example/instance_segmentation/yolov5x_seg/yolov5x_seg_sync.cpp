/**
 * @file yolov5x_seg_sync.cpp
 * @brief Yolov5xSegFactory synchronous inference example
 */

#include "factory/yolov5x_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov5xSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolov5xSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
