/**
 * @file yolox_s_leaky_async.cpp
 * @brief YOLOX_s_leaky asynchronous inference example
 */

#include "factory/yolox_s_leaky_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOX_s_leakyFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOX_s_leakyFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
