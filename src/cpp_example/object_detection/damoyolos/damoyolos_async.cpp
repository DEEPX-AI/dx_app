/**
 * @file damoyolos_async.cpp
 * @brief DAMO-YOLO-s asynchronous inference example
 */

#include "factory/damoyolos_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOsFactory>();
    dxapp::AsyncDetectionRunner<dxapp::DamoYOLOsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
