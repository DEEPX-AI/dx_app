/**
 * @file yolov5xs_wo_spp_512_async.cpp
 * @brief YOLOv5xs_wo_spp_512 asynchronous inference example
 */

#include "factory/yolov5xs_wo_spp_512_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5xs_wo_spp_512Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5xs_wo_spp_512Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
