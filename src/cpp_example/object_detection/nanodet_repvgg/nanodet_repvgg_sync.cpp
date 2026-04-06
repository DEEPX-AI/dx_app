/**
 * @file nanodet_repvgg_sync.cpp
 * @brief NanoDet synchronous inference example
 */

#include "factory/nanodet_repvgg_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDetFactory>();
    dxapp::SyncDetectionRunner<dxapp::NanoDetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
