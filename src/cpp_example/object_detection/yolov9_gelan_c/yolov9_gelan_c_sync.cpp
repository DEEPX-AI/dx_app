/**
 * @file yolov9_gelan_c_sync.cpp
 * @brief YOLOv9_gelan_c synchronous inference example
 */

#include "factory/yolov9_gelan_c_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv9_gelan_cFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOv9_gelan_cFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
