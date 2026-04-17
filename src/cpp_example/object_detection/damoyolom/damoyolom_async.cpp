/**
 * @file damoyolom_async.cpp
 * @brief DAMO-YOLO-m asynchronous inference example
 */

#include "factory/damoyolom_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOmFactory>();
    dxapp::AsyncDetectionRunner<dxapp::DamoYOLOmFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
