/**
 * @file yolo11l_seg_sync.cpp
 * @brief Yolo11lSegFactory synchronous inference example
 */

#include "factory/yolo11l_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11lSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo11lSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
