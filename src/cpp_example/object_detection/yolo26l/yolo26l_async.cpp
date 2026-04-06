/**
 * @file yolo26l_async.cpp
 * @brief Yolo26l asynchronous inference example
 */

#include "factory/yolo26l_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26lFactory>();
    dxapp::AsyncDetectionRunner<dxapp::Yolo26lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
