/**
 * @file yolov7_w6_wo_decoding_async.cpp
 * @brief YOLOv7_w6_wo_decoding asynchronous inference example
 */

#include "factory/yolov7_w6_wo_decoding_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_w6_wo_decodingFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv7_w6_wo_decodingFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
