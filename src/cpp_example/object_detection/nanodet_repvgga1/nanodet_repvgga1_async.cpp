/**
 * @file nanodet_repvgga1_async.cpp
 * @brief NanoDet_repvgga1 asynchronous inference example
 */

#include "factory/nanodet_repvgga1_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDet_repvgga1Factory>();
    dxapp::AsyncDetectionRunner<dxapp::NanoDet_repvgga1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
