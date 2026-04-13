/**
 * @file yolox_s_2_ppu_sync.cpp
 * @brief YOLOX_s_2_ppu synchronous inference example
 */

#include "factory/yolox_s_2_ppu_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOX_s_2_ppuFactory>();
    dxapp::SyncDetectionRunner<dxapp::YOLOX_s_2_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
