/**
 * @file yolo11s_seg_async.cpp
 * @brief Yolo11sSegFactory asynchronous inference example
 */

#include "factory/yolo11s_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11sSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolo11sSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
