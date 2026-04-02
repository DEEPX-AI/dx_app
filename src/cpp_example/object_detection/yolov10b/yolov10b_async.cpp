/**
 * @file yolov10b_async.cpp
 * @brief YOLOv10b asynchronous inference example
 */

#include "factory/yolov10b_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv10bFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv10bFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
