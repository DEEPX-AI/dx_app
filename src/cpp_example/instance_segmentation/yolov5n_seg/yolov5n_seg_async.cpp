/**
 * @file yolov5n_seg_async.cpp
 * @brief YOLOv5-Seg asynchronous inference example
 */

#include "factory/yolov5n_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5SegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::YOLOv5SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
