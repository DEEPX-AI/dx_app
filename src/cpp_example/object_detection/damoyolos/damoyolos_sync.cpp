/**
 * @file damoyolos_sync.cpp
 * @brief DAMO-YOLO-s synchronous inference example
 */

#include "factory/damoyolos_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOsFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLOsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
