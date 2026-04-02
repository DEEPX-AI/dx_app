/**
 * @file yolov8n_seg_sync.cpp
 * @brief YOLOv8-Seg synchronous instance segmentation example
 */

#include "factory/yolov8n_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8SegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::YOLOv8SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
