/**
 * @file yolox_l_leaky_sync.cpp
 * @brief YOLOX_l_leaky synchronous inference example
 */

#include "factory/yolox_l_leaky_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOX_l_leakyFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOX_l_leakyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
