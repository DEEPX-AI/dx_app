/**
 * @file yolov5x_seg_async.cpp
 * @brief Yolov5xSegFactory asynchronous inference example
 */

#include "factory/yolov5x_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov5xSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolov5xSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
