/**
 * @file damoyolom_sync.cpp
 * @brief DAMO-YOLO-m synchronous inference example
 */

#include "factory/damoyolom_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOmFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLOmFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
