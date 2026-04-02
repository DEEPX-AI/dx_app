/**
 * @file damoyolot_sync.cpp
 * @brief DAMO-YOLO-t synchronous inference example
 */

#include "factory/damoyolot_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOtFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLOtFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
