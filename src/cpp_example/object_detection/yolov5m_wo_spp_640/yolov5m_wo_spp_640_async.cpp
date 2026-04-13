/**
 * @file yolov5m_wo_spp_640_async.cpp
 * @brief YOLOv5m_wo_spp_640 asynchronous inference example
 */

#include "factory/yolov5m_wo_spp_640_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_wo_spp_640Factory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOv5m_wo_spp_640Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
