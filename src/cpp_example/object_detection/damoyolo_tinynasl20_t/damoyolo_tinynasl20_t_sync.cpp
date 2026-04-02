/**
 * @file damoyolo_tinynasl20_t_sync.cpp
 * @brief DAMO-YOLO-TinyNASL20-t synchronous inference example
 */

#include "factory/damoyolo_tinynasl20_t_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLO_tinynasl20_tFactory>();
    dxapp::SyncDetectionRunner<dxapp::DamoYOLO_tinynasl20_tFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
