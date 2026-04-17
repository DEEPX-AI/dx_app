/**
 * @file nanodet_repvgga1_sync.cpp
 * @brief NanoDet_repvgga1 synchronous inference example
 */

#include "factory/nanodet_repvgga1_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDet_repvgga1Factory>();
    dxapp::SyncDetectionRunner<dxapp::NanoDet_repvgga1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
