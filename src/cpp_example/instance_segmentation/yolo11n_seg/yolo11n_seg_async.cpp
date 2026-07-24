/**
 * @file yolo11n_seg_async.cpp
 * @brief Yolo11nSegFactory asynchronous inference example
 */

#include "factory/yolo11n_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11nSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolo11nSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
