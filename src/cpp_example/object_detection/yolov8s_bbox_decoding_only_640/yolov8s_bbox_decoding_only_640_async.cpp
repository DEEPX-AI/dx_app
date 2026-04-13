/**
 * @file yolov8s_bbox_decoding_only_640_async.cpp
 * @brief YOLOv8s_bbox_decoding_only_640 asynchronous inference example
 */

#include "factory/yolov8s_bbox_decoding_only_640_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv8s_bbox_decoding_only_640Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv8s_bbox_decoding_only_640Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
