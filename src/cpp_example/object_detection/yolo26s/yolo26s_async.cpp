/**
 * @file yolo26s_async.cpp
 * @brief Yolo26s asynchronous inference example
 */

#include "factory/yolo26s_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26sFactory>();
    dxapp::AsyncDetectionRunner<dxapp::Yolo26sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
