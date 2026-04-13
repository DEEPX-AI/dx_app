/**
 * @file yolov5s_4_async.cpp
 * @brief YOLOv5s_4 asynchronous inference example
 */

#include "factory/yolov5s_4_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5s_4Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5s_4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
