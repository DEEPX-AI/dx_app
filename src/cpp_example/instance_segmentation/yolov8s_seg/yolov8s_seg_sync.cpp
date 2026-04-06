/**
 * @file yolov8s_seg_sync.cpp
 * @brief YOLOv8-Seg synchronous instance segmentation example
 */

#include "factory/yolov8s_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8s_segFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::YOLOv8s_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
