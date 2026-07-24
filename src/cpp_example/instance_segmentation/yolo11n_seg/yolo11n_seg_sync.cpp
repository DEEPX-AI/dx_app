/**
 * @file yolo11n_seg_sync.cpp
 * @brief Yolo11nSegFactory synchronous inference example
 */

#include "factory/yolo11n_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11nSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo11nSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
