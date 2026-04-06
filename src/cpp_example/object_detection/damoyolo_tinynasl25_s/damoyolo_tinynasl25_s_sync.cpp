/**
 * @file damoyolo_tinynasl25_s_sync.cpp
 * @brief DAMO-YOLO-TinyNASL25-s synchronous inference example
 */

#include "factory/damoyolo_tinynasl25_s_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLO_tinynasl25_sFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLO_tinynasl25_sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
