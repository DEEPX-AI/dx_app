/**
 * @file yolov8x_seg_async.cpp
 * @brief Yolov8xSegFactory asynchronous inference example
 */

#include "factory/yolov8x_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8xSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolov8xSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
