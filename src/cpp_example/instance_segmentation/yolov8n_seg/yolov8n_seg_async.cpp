/**
 * @file yolov8n_seg_async.cpp
 * @brief YOLOv8Seg asynchronous instance segmentation example
 */

#include "factory/yolov8n_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8SegFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::YOLOv8SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
