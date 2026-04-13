/**
 * @file nanodet_repvgga12_async.cpp
 * @brief NanoDet_repvgga12 asynchronous inference example
 */

#include "factory/nanodet_repvgga12_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDet_repvgga12Factory>();
    dxapp::AsyncDetectionRunner<dxapp::NanoDet_repvgga12Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
