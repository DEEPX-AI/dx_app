/**
 * @file yolo11x_seg_async.cpp
 * @brief Yolo11xSegFactory asynchronous inference example
 */

#include "factory/yolo11x_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11xSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolo11xSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
