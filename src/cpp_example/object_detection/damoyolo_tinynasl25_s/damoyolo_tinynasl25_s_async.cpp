/**
 * @file damoyolo_tinynasl25_s_async.cpp
 * @brief DAMO-YOLO-TinyNASL25-s asynchronous inference example
 */

#include "factory/damoyolo_tinynasl25_s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLO_tinynasl25_sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::DamoYOLO_tinynasl25_sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
