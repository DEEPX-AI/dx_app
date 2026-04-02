/**
 * @file yolov11l_async.cpp
 * @brief YOLOv11l asynchronous inference example
 */

#include "factory/yolov11l_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv11lFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv11lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
