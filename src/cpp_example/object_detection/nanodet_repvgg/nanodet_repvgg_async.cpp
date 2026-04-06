/**
 * @file nanodet_repvgg_async.cpp
 * @brief NanoDet asynchronous inference example
 */

#include "factory/nanodet_repvgg_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDetFactory>();
    dxapp::AsyncDetectionRunner<dxapp::NanoDetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
