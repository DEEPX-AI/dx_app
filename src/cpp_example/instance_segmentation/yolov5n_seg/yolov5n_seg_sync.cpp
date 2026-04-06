/**
 * @file yolov5n_seg_sync.cpp
 * @brief YOLOv5-Seg synchronous inference example
 */

#include "factory/yolov5n_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5SegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::YOLOv5SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
