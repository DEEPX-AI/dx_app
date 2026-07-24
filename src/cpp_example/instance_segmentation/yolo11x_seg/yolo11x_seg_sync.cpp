/**
 * @file yolo11x_seg_sync.cpp
 * @brief Yolo11xSegFactory synchronous inference example
 */

#include "factory/yolo11x_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11xSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo11xSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
