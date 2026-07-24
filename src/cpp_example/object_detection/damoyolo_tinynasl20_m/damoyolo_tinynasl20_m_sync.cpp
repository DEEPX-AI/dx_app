/**
 * @file damoyolo_tinynasl20_m_sync.cpp
 * @brief DAMO-YOLO-TinyNASL20-m synchronous inference example
 */

#include "factory/damoyolo_tinynasl20_m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLO_tinynasl20_mFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLO_tinynasl20_mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
