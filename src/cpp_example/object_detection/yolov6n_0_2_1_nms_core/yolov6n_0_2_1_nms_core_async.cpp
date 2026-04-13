/**
 * @file yolov6n_0_2_1_nms_core_async.cpp
 * @brief YOLOv6n_0_2_1_nms_core asynchronous inference example
 */

#include "factory/yolov6n_0_2_1_nms_core_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv6n_0_2_1_nms_coreFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv6n_0_2_1_nms_coreFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
