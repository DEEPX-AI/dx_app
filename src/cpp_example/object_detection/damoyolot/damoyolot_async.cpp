/**
 * @file damoyolot_async.cpp
 * @brief DAMO-YOLO-t asynchronous inference example
 */

#include "factory/damoyolot_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOtFactory>();
    dxapp::AsyncDetectionRunner<dxapp::DamoYOLOtFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
