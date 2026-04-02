/**
 * @file damoyolol_async.cpp
 * @brief DAMO-YOLO-l asynchronous inference example
 */

#include "factory/damoyolol_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOlFactory>();
    dxapp::AsyncDetectionRunner<dxapp::DamoYOLOlFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
