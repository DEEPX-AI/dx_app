/**
 * @file yolov8s_seg_async.cpp
 * @brief YOLOv8s_seg asynchronous instance segmentation example
 */

#include "factory/yolov8s_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8s_segFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::YOLOv8s_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
