/**
 * @file yolov3_gluon_608_async.cpp
 * @brief YOLOv3_gluon_608 asynchronous inference example
 */

#include "factory/yolov3_gluon_608_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv3_gluon_608Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv3_gluon_608Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
