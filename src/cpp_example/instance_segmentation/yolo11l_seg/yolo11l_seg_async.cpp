/**
 * @file yolo11l_seg_async.cpp
 * @brief Yolo11lSegFactory asynchronous inference example
 */

#include "factory/yolo11l_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11lSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolo11lSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
