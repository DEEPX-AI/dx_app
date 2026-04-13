/**
 * @file yolo26n_seg_async.cpp
 * @brief YOLOv8Seg asynchronous instance segmentation example
 */

#include "factory/yolo26n_seg_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26n_segFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolo26n_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
