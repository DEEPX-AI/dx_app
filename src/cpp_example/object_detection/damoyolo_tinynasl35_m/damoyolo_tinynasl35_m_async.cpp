/**
 * @file damoyolo_tinynasl35_m_async.cpp
 * @brief DAMO-YOLO-TinyNASL35-m asynchronous inference example
 */

#include "factory/damoyolo_tinynasl35_m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DamoYOLO_tinynasl35_mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::DamoYOLO_tinynasl35_mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
