/**
 * @file yolo11m_seg_sync.cpp
 * @brief Yolo11mSegFactory synchronous inference example
 */

#include "factory/yolo11m_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo11mSegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo11mSegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
