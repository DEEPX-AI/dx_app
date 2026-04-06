/**
 * @file yolov5m_seg_sync.cpp
 * @brief YOLOv5-Seg synchronous inference example
 */

#include "factory/yolov5m_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_segFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::YOLOv5m_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
