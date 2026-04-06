/**
 * @file damoyolol_sync.cpp
 * @brief DAMO-YOLO-l synchronous inference example
 */

#include "factory/damoyolol_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLOlFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLOlFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
