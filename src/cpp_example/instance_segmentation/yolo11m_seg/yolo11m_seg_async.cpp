/**
 * @file yolo11m_seg_async.cpp
 * @brief Yolo11mSegFactory asynchronous inference example
 */

#include "factory/yolo11m_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11mSegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolo11mSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
