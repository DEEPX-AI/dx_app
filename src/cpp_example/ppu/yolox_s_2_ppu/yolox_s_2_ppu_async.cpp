/**
 * @file yolox_s_2_ppu_async.cpp
 * @brief YOLOX_s_2_ppu asynchronous inference example
 */

#include "factory/yolox_s_2_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOX_s_2_ppuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOX_s_2_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
