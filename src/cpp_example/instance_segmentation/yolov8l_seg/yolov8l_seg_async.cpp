/**
 * @file yolov8l_seg_async.cpp
 * @brief Yolov8lSegFactory asynchronous inference example
 */

#include "factory/yolov8l_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov8lSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolov8lSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
