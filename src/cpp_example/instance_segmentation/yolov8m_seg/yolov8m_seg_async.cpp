/**
 * @file yolov8m_seg_async.cpp
 * @brief YOLOv8m_seg asynchronous instance segmentation example
 */

#include "factory/yolov8m_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8m_segFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::YOLOv8m_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
