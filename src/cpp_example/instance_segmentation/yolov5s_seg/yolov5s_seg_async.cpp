/**
 * @file yolov5s_seg_async.cpp
 * @brief YOLOv5-Seg asynchronous inference example
 */

#include "factory/yolov5s_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_segFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::YOLOv5s_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
