/**
 * @file yolo26n_async.cpp
 * @brief Yolo26n asynchronous inference example
 */

#include "factory/yolo26n_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26nFactory>();
    dxapp::AsyncDetectionRunner<dxapp::Yolo26nFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
