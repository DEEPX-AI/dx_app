/**
 * @file yolo26x_async.cpp
 * @brief Yolo26x asynchronous inference example
 */

#include "factory/yolo26x_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26xFactory>();
    dxapp::AsyncDetectionRunner<dxapp::Yolo26xFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
