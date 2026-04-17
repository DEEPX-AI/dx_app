/**
 * @file yolox_s_leaky_sync.cpp
 * @brief YOLOX_s_leaky synchronous inference example
 */

#include "factory/yolox_s_leaky_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOX_s_leakyFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOX_s_leakyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
