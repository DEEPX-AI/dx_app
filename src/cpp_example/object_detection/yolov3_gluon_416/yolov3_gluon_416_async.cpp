/**
 * @file yolov3_gluon_416_async.cpp
 * @brief YOLOv3_gluon_416 asynchronous inference example
 */

#include "factory/yolov3_gluon_416_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3_gluon_416Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv3_gluon_416Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
