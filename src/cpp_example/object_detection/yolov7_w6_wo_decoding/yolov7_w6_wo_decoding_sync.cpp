/**
 * @file yolov7_w6_wo_decoding_sync.cpp
 * @brief YOLOv7_w6_wo_decoding synchronous inference example
 */

#include "factory/yolov7_w6_wo_decoding_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_w6_wo_decodingFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv7_w6_wo_decodingFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
