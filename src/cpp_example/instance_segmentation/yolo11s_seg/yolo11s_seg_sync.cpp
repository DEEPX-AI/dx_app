/**
 * @file yolo11s_seg_sync.cpp
 * @brief Yolo11sSegFactory synchronous inference example
 */

#include "factory/yolo11s_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11sSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo11sSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
