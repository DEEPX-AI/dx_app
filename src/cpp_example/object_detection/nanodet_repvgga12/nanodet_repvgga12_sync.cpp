/**
 * @file nanodet_repvgga12_sync.cpp
 * @brief NanoDet_repvgga12 synchronous inference example
 */

#include "factory/nanodet_repvgga12_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDet_repvgga12Factory>();
    dxapp::SyncDetectionRunner<dxapp::NanoDet_repvgga12Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
