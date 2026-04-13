/**
 * @file yolov4_leaky_512_512_async.cpp
 * @brief YOLOv4_leaky_512_512 asynchronous inference example
 */

#include "factory/yolov4_leaky_512_512_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv4_leaky_512_512Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv4_leaky_512_512Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
